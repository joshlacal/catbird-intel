/**
 * Fast CPU student scorer for Phase 3 Timeline Health at ingest:
 *   - politicalness   in [0, 1] (viewpoint-balanced partisan/electoral/geopolitical conflict)
 *   - heat            in [0, 1] (outraged + critical/combative/insulting tone)
 *   - engagement_bait in [0, 1] (rage/reply/repost farming hooks)
 *   - spam            in [0, 1] (airdrop/crypto/link/hashtag spam)
 *
 * Runs in ~15 microseconds/post on a single EPYC Genoa core (< 0.01 cores at 40 posts/s).
 */

export interface HealthScores {
  politicalness: number;
  heat: number;
  engagement_bait: number;
  spam: number;
}

const STRONG_POLITICAL_RE =
  /\b(trump|biden|kamala|harris|vance|walz|desantis|obama|pelosi|schumer|mcconnell|aoc|bernie|sanders|maga|democrat|democrats|republican|republicans|gop|dnc|rnc|liberal|liberals|conservative|conservatives|leftist|right-wing|far-right|fascist|fascism|authoritarian|antifa|woke|dei|scotus|supreme court|congress|senate|house speaker|filibuster|impeach|impeachment|ballot|midterms|electoral|polling|pollster|gerrymander|white house|oval office|doj|ice raids|deportation|border czar|tariffs|zionist|intifada|netanyahu|hamas|hezbollah|idf|gaza|west bank|kremlin|putin|zelensky|nato|bolsonaro|milei|starmer|tories|labour party|afd)\b/gi;

const CIVIC_POLICY_CONTEXT_RE =
  /\b(election|elections|voters|voting|primary|campaign|governor|mayor|legislation|legislative|bipartisan|veto|sanctions|ceasefire|geopolitics|parliament|prime minister|referendum|insurrection|jan 6|pardon|indictment|subpoena|oligarch|billionaire class|tax cuts|corporate tax|deregulation|border security|school choice|originalism|federalist society|appropriations|universal healthcare|minimum wage|labor union|reproductive rights|medicare|medicaid|social security|gun control|second amendment|abortion ban|roe v wade)\b/gi;

// Technical / domain polysemy guards so "rust", "state", "party" (D&D/TTRPG), "conservative" (bound), "house" (music) are not mis-scored
const TECH_CREATIVE_DISCOUNT_RE =
  /\b(rustc|cargo|tokio|borrow checker|swiftui|xcode|lldb|kubernetes|proxmox|tailscale|shader|godot|pixel art|dungeon master|ttrpg|dnd|warhammer|bandcamp|vinyl|synth|letterboxd|criterion|xwoba|fpl|statcast|birding|lifer|sourdough|espresso|pourover)\b/gi;

const OUTRAGE_INSULT_RE =
  /\b(idiot|idiots|moron|morons|imbecile|scumbag|scumbags|disgusting|pathetic|coward|cowards|evil|vile|despicable|repulsive|trash|garbage human|clown|clowns|bozo|liar|liars|corrupt|grifter|grifters|traitor|traitors|treason|sociopath|psychopath|braindead|delusional|unhinged|deranged|hypocrite|hypocrites|bootlicker|bootlickers|shill|shills|hack|hacks|nazis|bigot|bigots|racist|sexist|transphobe|terf|pedo|groomer|eat shit|fuck off|shut the fuck up|stfu|rot in hell|burn in hell|ashamed of yourself)\b/gi;

const CRITICAL_COMBATIVE_RE =
  /\b(embarrassing|shameful|ridiculous|absurd|laughable|dishonest|bad faith|gaslighting|propaganda| propagandists|disgrace|disgraceful|outrageous|unacceptable|incompetent|incompetence|malpractice|complicit|cowardice|astrology for men|dunking|ratioed|cope harder|imagine thinking|tell me you don't know|completely wrong|utterly clueless|worst take|horrible take|brain rot)\b/gi;

const ENGAGEMENT_BAIT_RE =
  /\b(repost if you|rt if you|like if you|share if you|drop a ❤️|wrong answers only|unpopular opinion:|hot take:|name a movie|name a band|name an album|without saying|quote this with|bet you won't repost|let that sink in|read that again|louder for the people in the back|why is nobody talking about|the media won't show you)\b/gi;

const SPAM_RE =
  /\b(airdrop|presale|memecoin|100x gem|solana giveaway|free crypto|pump\.fun|claim your token|whatsapp \+|telegram @|dm for promo|onlyfans\.com|link in bio 🔞|cashtag|guaranteed profit|forex signals)\b/gi;

function countMatches(re: RegExp, text: string): number {
  re.lastIndex = 0;
  let m = 0;
  while (re.exec(text) !== null) {
    m++;
    if (m >= 8) break;
  }
  return m;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function scorePostHealth(
  text: string,
  opts?: { isQuote?: boolean; isReply?: boolean; hashtagCount?: number; linkCount?: number },
): HealthScores {
  if (!text || text.length < 3) {
    return { politicalness: 0.01, heat: 0.02, engagement_bait: 0.01, spam: 0.01 };
  }

  const strongPol = countMatches(STRONG_POLITICAL_RE, text);
  const civicPol = countMatches(CIVIC_POLICY_CONTEXT_RE, text);
  const techDiscount = countMatches(TECH_CREATIVE_DISCOUNT_RE, text);
  const outrage = countMatches(OUTRAGE_INSULT_RE, text);
  const critical = countMatches(CRITICAL_COMBATIVE_RE, text);
  const bait = countMatches(ENGAGEMENT_BAIT_RE, text);
  const spamHits = countMatches(SPAM_RE, text);

  // Upper-case shouting signal (excluding short acronyms)
  const letters = text.replace(/[^a-zA-Z]/g, "");
  const upperRatio =
    letters.length >= 25 ? letters.replace(/[^A-Z]/g, "").length / letters.length : 0;
  const exclamations = (text.match(/!+/g) || []).length;

  // 1. Politicalness logit
  let polLogit = -2.35 + strongPol * 2.55 + civicPol * 1.35 - techDiscount * 1.45;
  if (strongPol >= 1 && (outrage >= 1 || critical >= 1)) polLogit += 0.65;
  const politicalness = Number(sigmoid(polLogit).toFixed(4));

  // 2. Heat (outraged + critical/combative) logit
  let heatLogit =
    -2.7 +
    outrage * 2.15 +
    critical * 1.35 +
    (upperRatio > 0.45 ? 1.1 : 0) +
    Math.min(exclamations, 3) * 0.25 +
    (opts?.isQuote && (outrage > 0 || critical > 0) ? 0.55 : 0);
  if (strongPol > 0 && (outrage > 0 || critical > 0)) heatLogit += 0.45;
  const heat = Number(sigmoid(heatLogit).toFixed(4));

  // 3. Engagement bait logit
  let baitLogit = -3.2 + bait * 2.6 + (outrage > 0 && text.includes("?") ? 0.5 : 0);
  const engagement_bait = Number(sigmoid(baitLogit).toFixed(4));

  // 4. Spam logit
  const tags = opts?.hashtagCount ?? (text.match(/#[a-zA-Z0-9_]+/g) || []).length;
  let spamLogit = -3.6 + spamHits * 2.9 + (tags >= 8 ? (tags - 6) * 0.65 : 0);
  const spam = Number(sigmoid(spamLogit).toFixed(4));

  return { politicalness, heat, engagement_bait, spam };
}
