"""Root's bounded, opt-in runner for one reviewed retained receipt fault.

Launch under a systemd control-group RuntimeMaxSec and an outer timeout. This
does not start a queue, publish an index, erase a charge, or remove a cache.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import uuid


ACCOUNT = '06165974855773273129'
WRITER = 'gce-capture-owner-20261021'
DB = Path('/data/gcs-drive-retained-authority/authority.sqlite')
OAUTH = Path('/root/.config/catbird-drive/oauth.json')
HEX32 = re.compile(r'[0-9a-f]{32}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bounded_file(path, maximum, private=False):
    require(not path.is_symlink(), 'symlinked_input')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                0 < info.st_size <= maximum, 'input_not_regular_and_bounded')
        if private:
            require(info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                    'input_not_private')
        raw = b''
        while len(raw) <= maximum:
            piece = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not piece:
                break
            raw += piece
        require(len(raw) == info.st_size, 'input_size_changed')
        return raw
    finally:
        os.close(fd)


class EventLog:
    """A newly created private, held-inode log; never reopen an existing file."""
    def __init__(self, path):
        self.path = path
        self.fd = self.parent_fd = None
        require(path.is_absolute() and str(path.parent.resolve()) == str(path.parent),
                'event_parent_not_canonical')
        self.parent_fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            info = os.fstat(self.parent_fd)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and
                    info.st_mode & 0o077 == 0,'event_parent_not_private')
            self.parent_identity = (info.st_dev,info.st_ino)
            self.fd = os.open(path.name,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,
                              0o600,dir_fd=self.parent_fd)
            info = os.fstat(self.fd)
            self.identity = (info.st_dev,info.st_ino)
            os.fsync(self.parent_fd)
        except BaseException:
            self.close()
            raise

    def append(self,value):
        info = os.fstat(self.fd)
        named = os.stat(self.path.name,dir_fd=self.parent_fd,follow_symlinks=False)
        parent = self.path.parent.lstat()
        require((parent.st_dev,parent.st_ino) == self.parent_identity and
                (named.st_dev,named.st_ino) == self.identity == (info.st_dev,info.st_ino) and
                stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                'event_identity_or_privacy_changed')
        raw = (json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()
        require(info.st_size+len(raw) <= 1<<20,'event_log_limit')
        while raw:
            written = os.write(self.fd,raw)
            require(written > 0,'event_write_failed')
            raw = raw[written:]
        os.fsync(self.fd)
        os.fsync(self.parent_fd)

    def close(self):
        for name in ('fd','parent_fd'):
            fd = getattr(self,name,None)
            if fd is not None:
                os.close(fd)
                setattr(self,name,None)


def held_pair_lock(path):
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info,named = os.fstat(fd),path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                info.st_uid == os.geteuid() and
                (info.st_dev,info.st_ino) == (named.st_dev,named.st_ino),
                'pair_lock_identity')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return os.fdopen(fd,'rb')
    except BaseException:
        os.close(fd)
        raise


class CrossPathPacedVerifier:
    """Gap from any prior synchronous write before a verifier's first GET."""
    def __init__(self, verifier):self.verifier=verifier
    def _call(self,name,*args,**kwargs):
        time.sleep(.60)
        return getattr(self.verifier,name)(*args,**kwargs)
    def verify_account(self,*args,**kwargs):return self._call('verify_account',*args,**kwargs)
    def verify_folder(self,*args,**kwargs):return self._call('verify_folder',*args,**kwargs)
    def verify_exact_payload_absent(self,*args,**kwargs):return self._call('verify_exact_payload_absent',*args,**kwargs)
    def verify_exact_receipt_absent(self,*args,**kwargs):return self._call('verify_exact_receipt_absent',*args,**kwargs)
    def gcs_digest(self,*args,**kwargs):return self._call('gcs_digest',*args,**kwargs)
    def drive_digest(self,*args,**kwargs):return self._call('drive_digest',*args,**kwargs)
    def close(self):return self.verifier.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--journal-sha256', required=True)
    p.add_argument('--external-facts', type=Path, required=True)
    p.add_argument('--external-sha256', required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute-receipt-recovery', action='store_true')
    mode.add_argument('--read-only-verifier-smoke', action='store_true')
    a = p.parse_args()
    require(os.geteuid() == 0, 'root_owner_required')
    require(a.execute_receipt_recovery or a.read_only_verifier_smoke, 'explicit_mode_required')
    require(HEX32.fullmatch(a.run_id) and HEX32.fullmatch(a.attempt_id), 'identity_format')
    require(all(HEX64.fullmatch(v) for v in (a.manifest_sha256,a.journal_sha256,a.external_sha256)), 'digest_format')
    now = time.time()
    require(math.isfinite(a.deadline_epoch) and now + 300 < a.deadline_epoch <= now + 900,
            'bounded_fresh_deadline_required')
    sys.path.insert(0,str(a.runtime))
    from gcs_drive_retained_continuous import local_writer_preflight
    from gcs_drive_retained_recovery_authority import RecoveryAuthority
    from gcs_drive_retained_receipt_recovery import ReceiptRecoveryController, finalize_reconciled_journal
    from gcs_drive_retained_overflow import pair_key, validate_overflow_manifest
    from gcs_drive_drain import SafeDrive
    from receipt_recovery_live_verifier import ReceiptRecoveryLiveVerifier

    recovery_id = uuid.uuid4().hex
    report = {'schema':'catbird.receipt-recovery-root-run.v1','recovery_id':recovery_id,
              'run_id':a.run_id,'attempt_id':a.attempt_id,'started_at':now,
              'deadline_epoch':a.deadline_epoch,'status':'started',
              'read_only_verifier_smoke':a.read_only_verifier_smoke}
    event_log = EventLog(a.events)
    event_log.append(report)
    authority = None
    verifier = None
    lock = None
    try:
        external_raw = bounded_file(a.external_facts,1<<20,private=True)
        require(hashlib.sha256(external_raw).hexdigest() == a.external_sha256,'external_digest')
        external = json.loads(external_raw)
        require(0 <= time.time()-external['observed_at'] < 300,'external_evidence_stale')
        require(external['human_no_other_uploads'] == 'No other uploads','owner_attestation_missing')
        for field in ('local_writer_processes','netcup_writer_processes','netcup_writer_timers','netcup_writer_units'):
            require(external[field] == [],'external_writer_present')
        local = local_writer_preflight()
        state = subprocess.run(['systemctl','is-active','catbird-drive-retained-continuous-20261021.service'],
                               capture_output=True,text=True,timeout=5).stdout.strip()
        require(state in ('inactive','failed'),'continuous_controller_not_inactive')
        require(subprocess.run(['systemctl','is-active','--quiet','catbird-gcs-buffer.service'],timeout=5).returncode == 0,
                'capture_not_active')
        require(shutil.disk_usage('/data').free > 103*(1<<30),'capture_disk_floor')
        capture = json.loads(bounded_file(Path('/data/gcs-buffer/status.json'),1<<20))
        require(capture['reason'] in ('ready','uploading') and 0 <= time.time()-capture['updated_at_unix'] < 120,
                'capture_status_not_fresh')
        work = Path('/data/gcs-drive-retained-'+a.run_id)
        require(work.is_dir() and not work.is_symlink(),'work_directory_missing')
        require(str(work.resolve()) == str(work),'work_directory_not_canonical')
        lock = held_pair_lock(work/'overflow.lock')
        journal_raw = bounded_file(work/'journal.json',1<<20,private=True)
        require(hashlib.sha256(journal_raw).hexdigest() == a.journal_sha256,'journal_digest')
        c = sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True)
        c.row_factory = sqlite3.Row
        try:
            c.execute('BEGIN')
            run = dict(c.execute('SELECT * FROM retained_run WHERE id=?',(a.run_id,)).fetchone())
            old = dict(c.execute('SELECT * FROM retained_attempt WHERE id=?',(a.attempt_id,)).fetchone())
            account = dict(c.execute('SELECT * FROM retained_account WHERE id=1').fetchone())
        finally:
            c.close()
        manifest_raw = bytes(run['manifest_raw'])
        require(hashlib.sha256(manifest_raw).hexdigest() == a.manifest_sha256 == run['manifest_sha256'], 'manifest_digest')
        manifest = json.loads(manifest_raw)
        validate_overflow_manifest(manifest,release_id=a.run_id,account_id=ACCOUNT)
        key = pair_key(manifest)
        require(key == run['pair_key'] and run['status'] == 'active','active_pair_identity')
        entry = json.loads(journal_raw)['segments'][key]
        receipt_session = entry['drive']['receipt'].get('session')
        payload_session = entry['drive']['payload'].get('session')
        is_payload_tail = receipt_session is None and isinstance(payload_session, str)
        if is_payload_tail:
            require(old['run_id'] == a.run_id and old['status'] == 'unresolved' and old['http_status'] == 500 and
                    0 < old['bytes'] <= min(8 << 20, manifest['payload']['bytes']) and
                    time.time() > old['charge_through'], 'original_payload_tail_fault')
            session = payload_session
            permitted_branch = 'payload_tail_and_receipt_auto'
        else:
            require(old['run_id'] == a.run_id and old['status'] == 'unresolved' and old['http_status'] == 500 and
                    old['bytes'] == manifest['receipt']['bytes'] and time.time() > old['charge_through'],'original_receipt_fault')
            session = receipt_session
            permitted_branch = 'receipt_only_auto'
        require(account['process_pid'] and not Path('/proc/'+str(account['process_pid'])).exists(),'previous_writer_not_dead')
        require(isinstance(session, str) and hashlib.sha256(session.encode()).hexdigest() == old['target_sha256'],'original_session_digest')
        drive_ids = {kind:entry['drive'][kind]['id'] for kind in ('folder','payload','receipt')}
        verifier = CrossPathPacedVerifier(ReceiptRecoveryLiveVerifier(
            manifest=manifest,entry_digests={kind:entry[kind+'_digest'] for kind in ('payload','receipt')},
            drive_ids=drive_ids,payload_name=entry['payload_name'],receipt_name=entry['receipt_name'],
            oauth_file=OAUTH,deadline_epoch=a.deadline_epoch))
        verifier.verify_account(ACCOUNT)
        verifier.verify_folder(drive_ids['folder'],manifest['folder_name'],manifest['drive_root'])
        if a.read_only_verifier_smoke:
            digests = {}
            for kind in ('payload','receipt'):
                ref = manifest[kind]
                digests['gcs_'+kind] = verifier.gcs_digest(ref['object'],ref['generation'])
            if not is_payload_tail:
                digests['drive_payload'] = verifier.drive_digest(drive_ids['payload'])
            absent = verifier.verify_exact_receipt_absent(account_id=ACCOUNT,root_id=manifest['drive_root'],
                folder_id=drive_ids['folder'],receipt_id=drive_ids['receipt'])
            report.update(status='read_only_verifier_smoke_pass',digests=digests,receipt_absent=absent,
                          ledger_writes=0,cloud_writes=0,finished_at=time.time())
            event_log.append(report)
            print(json.dumps(report,sort_keys=True))
            return 0
        # Cross-path gap: verifier GETs and charged SafeDrive writes share one
        # synchronous owner; wait from the completed final verifier request.
        time.sleep(.60)
        evidence = {
            'schema':'catbird.drive.receipt-recovery-review.v1',
            'account_id':ACCOUNT,'writer_id':WRITER,'original_run_id':a.run_id,
            'original_attempt_id':a.attempt_id,'receipt_id':drive_ids['receipt'],
            'session_sha256':old['target_sha256'],'manifest_sha256':a.manifest_sha256,
            'pair_key':key,'folder_id':drive_ids['folder'],'payload_id':drive_ids['payload'],
            'receipt_name':entry['receipt_name'],'receipt_stage_path':str(work/'stage'/key/entry['receipt_name']),
            'receipt_sha256':manifest['receipt']['sha256'],
            'account_identity_verified':True,'all_writers_excluded':True,
            'exclusion_valid_until':a.deadline_epoch,'observed_at':time.time(),
            'owner_reference':'Root sole writer under 2026-10-21-network-capture-resume; exact receipt HTTP500 recovery',
            'external_facts_sha256':a.external_sha256,'local_writer_preflight':local,
            'permitted_branch':permitted_branch,
        }
        if is_payload_tail:
            evidence.update(
                payload_name=entry['payload_name'],
                payload_stage_path=str(work/'stage'/key/entry['payload_name']),
                payload_sha256=manifest['payload']['sha256'],
            )
        require(hashlib.sha256(bounded_file(work/'journal.json',1<<20,private=True)).hexdigest() == a.journal_sha256,
                'journal_changed_before_claim')
        require(0 <= time.time()-external['observed_at'] < 300,'external_evidence_stale_before_claim')
        authority = RecoveryAuthority.open_new(DB,recovery_id=recovery_id,original_run_id=a.run_id,
            original_attempt_id=a.attempt_id,account_id=ACCOUNT,writer_id=WRITER,
            receipt_id=drive_ids['receipt'],session_url=session,evidence=evidence,deadline_at=a.deadline_epoch)
        report['recovery_id'] = authority.release_id
        drive = SafeDrive(OAUTH,authority)
        def paced_drive_request(*args,**kwargs):
            time.sleep(.60)
            return drive.request(*args,**kwargs)
        controller = ReceiptRecoveryController(manifest_raw=manifest_raw,journal_raw=journal_raw,
            original_run_id=a.run_id,original_attempt_id=a.attempt_id,account_id=ACCOUNT,writer_id=WRITER,
            work_dir=work,authority=authority,drive_request=paced_drive_request,verifier=verifier)
        event_log.append({**report,'status':'claimed','observed_at':time.time(),
                               'manifest_sha256':a.manifest_sha256,'journal_sha256':a.journal_sha256})
        result = controller.run_authorized(operator_reference=evidence['owner_reference'])
        if result['state'] == 'forensically_verified':
            transition = finalize_reconciled_journal(controller,held_lock_fd=lock.fileno())
            require(transition['state'] == 'drive_verified_no_send','journal_transition_incomplete')
            report.update(status='journal_reconciled_pending_index',result=result,
                          journal_transition=transition,finished_at=time.time())
        else:
            report.update(status=result['state'],result=result,finished_at=time.time())
        event_log.append(report)
        print(json.dumps(report,sort_keys=True))
        return 0 if report['status'] == 'journal_reconciled_pending_index' else 2
    except Exception as error:
        report.update(status='failed_preserved',error_type=type(error).__name__,finished_at=time.time())
        event_log.append(report)
        print(json.dumps(report,sort_keys=True))
        return 1
    finally:
        if authority is not None:authority.close()
        if verifier is not None:verifier.close()
        if lock is not None:lock.close()
        event_log.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status':'preflight_failed','error_type':type(error).__name__}),file=sys.stderr)
        raise SystemExit(1)
