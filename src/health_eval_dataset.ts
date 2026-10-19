export interface EvalItem {
  id: string;
  text: string;
  viewpoint: "left" | "right" | "civic" | "non_political_trap" | "non_political";
  is_political: 0 | 1;
  is_heated: 0 | 1;
  isQuote?: boolean;
}

/**
 * Held-out viewpoint-balanced human-checked benchmark (60 items):
 * - 12 left/progressive political (6 calm/analytic, 6 heated/combative)
 * - 12 right/conservative political (6 calm/analytic, 6 heated/combative)
 * - 8 civic/electoral/geopolitical policy (5 calm, 3 heated)
 * - 16 polysemy trap non-political posts ("state", "conservative bound", "party", "cabinet", "house", "rust", "macro", "border")
 * - 12 everyday subculture posts (8 calm, 4 non-political high-heat posts to test orthogonal Heat vs Politics)
 */
export const VIEWPOINT_BALANCED_EVAL_SET: EvalItem[] = [
  // Left / progressive — calm
  { id: "L1", viewpoint: "left", is_political: 1, is_heated: 0, text: "Universal healthcare and expanding Medicare coverage would lower per-capita costs across the Senate budget projection." },
  { id: "L2", viewpoint: "left", is_political: 1, is_heated: 0, text: "Progressive Democrats in Congress are pushing for stronger labor union protections and a higher federal minimum wage bill." },
  { id: "L3", viewpoint: "left", is_political: 1, is_heated: 0, text: "Bernie Sanders and AOC outlined a new climate transition framework ahead of the Democratic primary debates." },
  { id: "L4", viewpoint: "left", is_political: 1, is_heated: 0, text: "Supreme Court reproductive rights rulings after Roe v Wade continue to reshape state ballot measures and voter turnout." },
  { id: "L5", viewpoint: "left", is_political: 1, is_heated: 0, text: "Taxing billionaire class wealth gains and closing corporate loopholes is central to the DNC platform this cycle." },
  { id: "L6", viewpoint: "left", is_political: 1, is_heated: 0, text: "Public defenders and civil rights advocates filed an amicus brief challenging the DOJ deportation policy in federal court." },

  // Left / progressive — heated / combative
  { id: "L7", viewpoint: "left", is_political: 1, is_heated: 1, isQuote: true, text: "MAGA Republicans in Congress are spineless cowards and corrupt grifters destroying Medicaid for billionaire tax cuts!" },
  { id: "L8", viewpoint: "left", is_political: 1, is_heated: 1, text: "What an utterly delusional, dishonest fascist propaganda take from right-wing pundits defending Trump's tariffs." },
  { id: "L9", viewpoint: "left", is_political: 1, is_heated: 1, text: "Senate leadership is pathetic and complicit—total cowardice watching GOP extremists gut voting rights." },
  { id: "L10", viewpoint: "left", is_political: 1, is_heated: 1, text: "These MAGA clowns and bootlickers are completely unhinged liars. Embarrassing disgrace to the Supreme Court." },
  { id: "L11", viewpoint: "left", is_political: 1, is_heated: 1, text: "Imagine thinking billionaire oligarchs care about working voters. Utterly clueless bad faith garbage from DNC consultants too." },
  { id: "L12", viewpoint: "left", is_political: 1, is_heated: 1, text: "Every single GOP senator who voted for this vile, despicable deportation bill should be ashamed of yourself." },

  // Right / conservative — calm
  { id: "R1", viewpoint: "right", is_political: 1, is_heated: 0, text: "Conservative voters in the Republican primary are prioritizing border security, deregulation, and lower corporate tax rates." },
  { id: "R2", viewpoint: "right", is_political: 1, is_heated: 0, text: "The RNC and GOP governors argued that Second Amendment protections and school choice vouchers enjoy broad legislative support." },
  { id: "R3", viewpoint: "right", is_political: 1, is_heated: 0, text: "Trump's trade advisor outlined the administration's reciprocal tariffs strategy and manufacturing reshoring goals at the White House." },
  { id: "R4", viewpoint: "right", is_political: 1, is_heated: 0, text: "Federalist Society scholars debated originalism and Supreme Court precedent on administrative agency deference." },
  { id: "R5", viewpoint: "right", is_political: 1, is_heated: 0, text: "House Republicans introduced a balanced budget amendment and spending caps ahead of the congressional appropriations vote." },
  { id: "R6", viewpoint: "right", is_political: 1, is_heated: 0, text: "Polling in midterms battleground districts shows independent voters shifting toward conservative fiscal legislation." },

  // Right / conservative — heated / combative
  { id: "R7", viewpoint: "right", is_political: 1, is_heated: 1, isQuote: true, text: "Woke leftist Democrats and Pelosi hacks are delusional hypocrites pushing lunatic DEI propaganda on taxpayers!" },
  { id: "R8", viewpoint: "right", is_political: 1, is_heated: 1, text: "Biden and Kamala's incompetent border czar policy is a shameful, outrageous disgrace run by total clowns." },
  { id: "R9", viewpoint: "right", is_political: 1, is_heated: 1, text: "Leftist media shills and DNC liars gaslighting voters again—pathetic, dishonest frauds!" },
  { id: "R10", viewpoint: "right", is_political: 1, is_heated: 1, text: "These woke bureaucrats are corrupt, braindead morons destroying small businesses with absurd regulations." },
  { id: "R11", viewpoint: "right", is_political: 1, is_heated: 1, text: "Liberal governors acting like authoritarian hypocrites while cities crumble. Utterly ridiculous incompetence!" },
  { id: "R12", viewpoint: "right", is_political: 1, is_heated: 1, text: "RNC establishment cowards caving to Schumer and the Democrats again—bunch of spineless traitors." },

  // Civic / geopolitical / electoral
  { id: "C1", viewpoint: "civic", is_political: 1, is_heated: 0, text: "NATO foreign ministers met in Brussels to discuss Ukraine sanctions, Kremlin diplomacy, and European defense budgets." },
  { id: "C2", viewpoint: "civic", is_political: 1, is_heated: 0, text: "UN negotiators proposed a new ceasefire framework in Gaza following talks between Israeli and regional diplomats." },
  { id: "C3", viewpoint: "civic", is_political: 1, is_heated: 0, text: "Early ballot returns and county election precinct data suggest higher turnout in the gubernatorial primary." },
  { id: "C4", viewpoint: "civic", is_political: 1, is_heated: 0, text: "The Labour Party and Tories debated parliament's new housing legislation and Prime Minister Starmer's budget." },
  { id: "C5", viewpoint: "civic", is_political: 1, is_heated: 0, text: "Bipartisan Senate committee subpoenaed records on federal procurement and gerrymandering redistricting maps." },
  { id: "C6", viewpoint: "civic", is_political: 1, is_heated: 1, text: "Putin's Kremlin apologists are vile, despicable propagandists spreading disgusting lies about war crimes." },
  { id: "C7", viewpoint: "civic", is_political: 1, is_heated: 1, text: "Netanyahu and Hamas hardliners both fueling endless bloodshed—horrific, shameful malpractice by every leader involved!" },
  { id: "C8", viewpoint: "civic", is_political: 1, is_heated: 1, text: "What an absurd, incompetent election pollster take—imagine thinking a 2-point Crosstab is a landslide, total clown show." },

  // Polysemy traps — non-political posts containing words like "conservative", "state", "party", "house", "cabinet", "border"
  { id: "T1", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "In rustc and tokio, a conservative upper bound on channel buffer size prevents borrow checker state lifetime issues." },
  { id: "T2", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Our D&D TTRPG party finally defeated the dungeon master's campaign boss after a three-hour initiative roll." },
  { id: "T3", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Spinning 90s Chicago house vinyl and deep techno records on the turntable all afternoon #bandcamp" },
  { id: "T4", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Built a custom walnut kitchen cabinet with dovetail joints and brass hardware in the woodshop today." },
  { id: "T5", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Added a 2px CSS border radius and SwiftUI state binding to the macOS window toolbar." },
  { id: "T6", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "My golden retriever lab mix loves chasing tennis balls across the state park trail every morning." },
  { id: "T7", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Dialing in a 18g espresso shot on the lever machine—light roast Ethiopian pourover notes of jasmine and bergamot." },
  { id: "T8", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Shot two rolls of Kodak Portra 400 with a 90mm macro lens on beetles and wildflowers in the garden." },
  { id: "T9", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Migrated my Proxmox homelab Kubernetes cluster to Tailscale wireguard mesh and ZFS mirrored NVMe drives." },
  { id: "T10", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "His xwOBA and barrel rate against four-seam fastballs in Statcast are elite over a 200 plate appearance sample." },
  { id: "T11", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Godot 4.3 GDScript shader compilation stutter is fixed when pre-warming the Vulkan pipeline cache." },
  { id: "T12", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Spotted a lifer cerulean warbler during spring migration birding along the ridge!" },
  { id: "T13", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Taylor Swift's acoustic surprise song mashup on the Eras Tour last night had the whole stadium singing." },
  { id: "T14", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "Playing Rust wipe day on a vanilla trio server—built our stone 2x2 base near launch site." },
  { id: "T15", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "FPL managers captaining Salah in a double gameweek while Arsenal's low block xG conceded stays under 0.6." },
  { id: "T16", viewpoint: "non_political_trap", is_political: 0, is_heated: 0, text: "High-ratio sourdough starter at 78% hydration gave the best open crumb and blistered crust yet." },

  // Everyday non-political (including 4 heated non-political posts to test orthogonal Heat vs Politics!)
  { id: "N1", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "Finished reading Ursula K. Le Guin's The Left Hand of Darkness—what a masterpiece of worldbuilding." },
  { id: "N2", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "HRRR mesoscale model run shows a strong 500mb trough and lake-effect snow bands setting up tonight." },
  { id: "N3", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "AEW Dynamite main event match between Ospreay and Danielson had incredible ring psychology." },
  { id: "N4", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "New protected bike lane and 5-minute BRT bus headways cut my morning commute in half." },
  { id: "N5", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "Lubed Gazzew Boba U4T tactile switches on an Alice layout mechanical keyboard with fountain pen ink notes." },
  { id: "N6", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "JWST NIRCam deep field image of high-redshift galaxies is breathtaking." },
  { id: "N7", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "Criterion Channel's 4K restoration of Paris, Texas looks gorgeous on OLED." },
  { id: "N8", viewpoint: "non_political", is_political: 0, is_heated: 0, text: "Quantized Llama-3 8B to Q4_K_M GGUF running at 95 tokens/sec on Apple Silicon Metal." },
  // Non-political BUT high heat (crucial test that heat is separate from politics!)
  { id: "N9", viewpoint: "non_political", is_political: 0, is_heated: 1, isQuote: true, text: "What an utterly clueless, braindead moron take on compiler optimization—embarrassing, dishonest garbage!" },
  { id: "N10", viewpoint: "non_political", is_political: 0, is_heated: 1, text: "These toxic quote-dunking clowns and grifters are pathetic scumbags. Shut the fuck up and touch grass!" },
  { id: "N11", viewpoint: "non_political", is_political: 0, is_heated: 1, text: "Imagine thinking this ridiculous, laughable scam is art—utterly delusional bad faith trash." },
  { id: "N12", viewpoint: "non_political", is_political: 0, is_heated: 1, text: "The refereeing in tonight's game was outrageous, shameful incompetence! Absolute disgrace!" },
];
