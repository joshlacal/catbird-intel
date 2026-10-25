// transcribe.swift: fully on-device speech-to-text for the labeling tool.
//
// Usage:
//   transcribe <audio-file>                 transcribe (en-US) and print one JSON object
//   transcribe --engine sfspeech <file>     force the SFSpeechRecognizer on-device fallback
//   transcribe --install-assets             download/install Apple's on-device en-US
//                                           SpeechTranscriber model (one-time, no audio sent)
//   transcribe --status                     print engine/asset availability as JSON
//
// Output (stdout, exactly one JSON object):
//   {"text": str, "engine": "SpeechTranscriber"|"SFSpeechRecognizer-ondevice",
//    "locale": "en-US", "ms": int}
//   or {"error": str} with a non-zero exit status.
//
// Privacy: audio and transcripts never leave the Mac.
//   * SpeechTranscriber/SpeechAnalyzer (macOS 26+) runs entirely on-device; its model
//     is an Apple asset installed locally. We never install assets implicitly during a
//     transcription: if the model is missing we fail and point at --install-assets.
//   * The SFSpeechRecognizer fallback is used only when the recognizer reports
//     supportsOnDeviceRecognition, and the request sets requiresOnDeviceRecognition.
//     If on-device recognition is unsupported we refuse instead of letting the
//     framework fall back to Apple's servers.

import AVFAudio
import Foundation
import Speech

let localeIdentifier = "en-US"

struct CLIError: Error, CustomStringConvertible {
  let description: String
  let code: Int32
  init(_ description: String, code: Int32 = 1) {
    self.description = description
    self.code = code
  }
}

func emit(_ object: [String: Any]) {
  let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
    ?? Data("{\"error\": \"failed to encode output\"}".utf8)
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data("\n".utf8))
}

func elapsedMs(since start: ContinuousClock.Instant) -> Int {
  let d = ContinuousClock.now - start
  return Int(d.components.seconds * 1000 + d.components.attoseconds / 1_000_000_000_000_000)
}

// MARK: - SpeechTranscriber (preferred)

enum TranscriberReadiness {
  case ready(SpeechTranscriber)
  /// Hardware/OS/locale cannot use SpeechTranscriber at all: fall back is acceptable.
  case unavailable(String)
  /// Supported, but the on-device model is not installed yet.
  case assetsMissing(String)
}

func makeTranscriber() async -> (Locale?, SpeechTranscriber?, String?) {
  guard SpeechTranscriber.isAvailable else {
    return (nil, nil, "SpeechTranscriber.isAvailable is false on this device")
  }
  guard let locale = await SpeechTranscriber.supportedLocale(
    equivalentTo: Locale(identifier: localeIdentifier))
  else {
    return (nil, nil, "SpeechTranscriber does not support \(localeIdentifier)")
  }
  return (locale, SpeechTranscriber(locale: locale, preset: .transcription), nil)
}

func transcriberReadiness() async -> TranscriberReadiness {
  let (_, transcriber, reason) = await makeTranscriber()
  guard let transcriber else { return .unavailable(reason ?? "unavailable") }
  switch await AssetInventory.status(forModules: [transcriber]) {
  case .installed:
    return .ready(transcriber)
  case .unsupported:
    return .unavailable("AssetInventory reports the \(localeIdentifier) model as unsupported")
  case .supported:
    return .assetsMissing("on-device \(localeIdentifier) speech model is not installed")
  case .downloading:
    return .assetsMissing("on-device \(localeIdentifier) speech model is still downloading")
  @unknown default:
    return .unavailable("unknown AssetInventory status")
  }
}

func installAssets() async throws -> String {
  let (_, transcriber, reason) = await makeTranscriber()
  guard let transcriber else { throw CLIError(reason ?? "SpeechTranscriber unavailable") }
  if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
    try await request.downloadAndInstall()
  }
  let status = await AssetInventory.status(forModules: [transcriber])
  guard status == .installed else {
    throw CLIError("asset install finished but status is \(status)")
  }
  return "installed"
}

func transcribeWithSpeechTranscriber(_ transcriber: SpeechTranscriber, url: URL) async throws
  -> String
{
  let file = try AVAudioFile(forReading: url)
  let analyzer = SpeechAnalyzer(modules: [transcriber])

  async let collected: [String] = {
    var parts: [String] = []
    for try await result in transcriber.results where result.isFinal {
      parts.append(String(result.text.characters))
    }
    return parts
  }()

  if let lastSample = try await analyzer.analyzeSequence(from: file) {
    try await analyzer.finalizeAndFinish(through: lastSample)
  } else {
    await analyzer.cancelAndFinishNow()
  }
  let parts = try await collected
  return parts.joined().trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - SFSpeechRecognizer on-device fallback

// We deliberately never call SFSpeechRecognizer.requestAuthorization(). For a CLI, TCC
// attributes the request to the *responsible* process (Terminal, an IDE, or the agent
// host), which has no NSSpeechRecognitionUsageDescription, so tccd kills us with
// SIGABRT (TCC_CRASHING_DUE_TO_PRIVACY_VIOLATION) before any prompt is shown. Starting
// a recognition task while undetermined trips the same abort. So: authorized or refuse.
// SpeechTranscriber (the default engine) needs no Speech Recognition grant at all.
func checkSpeechAuthorization() throws {
  let status = SFSpeechRecognizer.authorizationStatus()
  guard status == .authorized else {
    throw CLIError(
      "SFSpeechRecognizer fallback not authorized (TCC status \(status.rawValue)); a CLI "
        + "cannot raise the Speech Recognition prompt. Use the default SpeechTranscriber "
        + "engine (no grant needed).", code: 4)
  }
}

func transcribeWithSFSpeech(url: URL) async throws -> String {
  guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier)) else {
    throw CLIError("SFSpeechRecognizer does not support \(localeIdentifier)")
  }
  // Privacy gate: refuse rather than allow any server-side recognition path.
  guard recognizer.supportsOnDeviceRecognition else {
    throw CLIError(
      "refusing: SFSpeechRecognizer on-device recognition is unsupported for "
        + "\(localeIdentifier); server recognition is never allowed")
  }
  try checkSpeechAuthorization()
  let request = SFSpeechURLRecognitionRequest(url: url)
  request.requiresOnDeviceRecognition = true
  request.shouldReportPartialResults = false
  request.addsPunctuation = true
  guard request.requiresOnDeviceRecognition else {
    throw CLIError("refusing: could not force on-device recognition")
  }

  return try await withCheckedThrowingContinuation { cont in
    var resumed = false
    let lock = NSLock()
    func finish(_ r: Result<String, Error>) {
      lock.lock()
      defer { lock.unlock() }
      if resumed { return }
      resumed = true
      cont.resume(with: r)
    }
    _ = recognizer.recognitionTask(with: request) { result, error in
      if let error {
        finish(.failure(CLIError("SFSpeechRecognizer: \(error.localizedDescription)")))
      } else if let result, result.isFinal {
        finish(.success(result.bestTranscription.formattedString))
      }
    }
  }
}

// MARK: - Entry point

@main
struct Transcribe {
  static func main() async {
    var args = Array(CommandLine.arguments.dropFirst())
    do {
      if args == ["--install-assets"] {
        let start = ContinuousClock.now
        let state = try await installAssets()
        emit(["assets": state, "locale": localeIdentifier, "ms": elapsedMs(since: start)])
        exit(0)
      }
      if args == ["--status"] {
        let readiness = await transcriberReadiness()
        var out: [String: Any] = ["locale": localeIdentifier]
        switch readiness {
        case .ready: out["speechTranscriber"] = "ready"
        case .unavailable(let r): out["speechTranscriber"] = "unavailable: \(r)"
        case .assetsMissing(let r): out["speechTranscriber"] = "assets-missing: \(r)"
        }
        let sf = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
        out["sfSpeechOnDeviceSupported"] = sf?.supportsOnDeviceRecognition ?? false
        out["sfSpeechAuthorization"] = SFSpeechRecognizer.authorizationStatus().rawValue
        emit(out)
        exit(0)
      }

      var engine = "auto"
      if args.count >= 2, args[0] == "--engine" {
        engine = args[1]
        args.removeFirst(2)
      }
      guard args.count == 1 else {
        throw CLIError(
          "usage: transcribe [--engine auto|speechtranscriber|sfspeech] <audio-file> | "
            + "--install-assets | --status", code: 2)
      }
      guard ["auto", "speechtranscriber", "sfspeech"].contains(engine) else {
        throw CLIError("unknown engine \(engine)", code: 2)
      }
      let url = URL(fileURLWithPath: args[0])
      guard FileManager.default.isReadableFile(atPath: url.path) else {
        throw CLIError("audio file not readable: \(url.path)", code: 2)
      }

      let start = ContinuousClock.now
      var text: String
      var engineName: String
      if engine == "sfspeech" {
        text = try await transcribeWithSFSpeech(url: url)
        engineName = "SFSpeechRecognizer-ondevice"
      } else {
        switch await transcriberReadiness() {
        case .ready(let transcriber):
          text = try await transcribeWithSpeechTranscriber(transcriber, url: url)
          engineName = "SpeechTranscriber"
        case .assetsMissing(let reason):
          // Never download implicitly mid-transcription; make the one-time install explicit.
          throw CLIError("\(reason); run `transcribe --install-assets` once", code: 3)
        case .unavailable(let reason):
          if engine == "speechtranscriber" { throw CLIError(reason) }
          text = try await transcribeWithSFSpeech(url: url)
          engineName = "SFSpeechRecognizer-ondevice"
        }
      }
      emit([
        "text": text, "engine": engineName, "locale": localeIdentifier,
        "ms": elapsedMs(since: start),
      ])
      exit(0)
    } catch let e as CLIError {
      emit(["error": e.description])
      exit(e.code)
    } catch {
      emit(["error": "\(type(of: error)): \(error.localizedDescription)"])
      exit(1)
    }
  }
}
