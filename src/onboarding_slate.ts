/**
 * The mix a scene's first twelve positions aim for: member tiers by follower count, and the slate
 * that arranges roles and tiers over the front of an ordered list. Pure functions of their inputs.
 */
import type { CommunityRole, IntelMemberEntry } from "./intel_catalog.ts";
import type { LiveOnboardingConfig, MemberTier, OrderedMember } from "./onboarding_live_types.ts";

interface SlateSlot {
  roles?: readonly CommunityRole[];
  tier?: MemberTier;
}

/** The kinds of account the first six positions aim for, repeated for positions seven to twelve. */
const SLATE: readonly SlateSlot[] = [
  { roles: ["anchor_creator"] },
  { roles: ["visual_maker"] },
  { roles: ["conversationalist"] },
  { tier: "rising" },
  { roles: ["anchor_creator", "project_or_studio"] },
  { roles: ["scene_curator"] },
];
const SLATE_POSITIONS = SLATE.length * 2;

export function tierOf(member: IntelMemberEntry, config: LiveOnboardingConfig): MemberTier {
  if (member.followers >= config.headlinerFollowers) return "headliner";
  if (member.followers < config.risingFollowers) return "rising";
  return "core";
}

/** Gives the first twelve positions a mix of account kinds and at most a few very large accounts. */
export function applySlate(byKey: OrderedMember[], config: LiveOnboardingConfig): OrderedMember[] {
  const remaining = [...byKey];
  const ordered: OrderedMember[] = [];
  let headliners = 0;
  let megas = 0;
  const withinCaps = (c: OrderedMember, position: number): boolean => {
    const headlinerCap = config.maxHeadlinersPerSix * (Math.floor(position / SLATE.length) + 1);
    if (c.tier === "headliner" && headliners >= headlinerCap) return false;
    if (c.member.followers > config.megaFollowers && megas >= 1) return false;
    return true;
  };
  for (let position = 0; position < SLATE_POSITIONS && remaining.length > 0; position++) {
    const slot = SLATE[position % SLATE.length]!;
    const wanted = (c: OrderedMember): boolean =>
      (slot.tier ? c.tier === slot.tier : true) && (slot.roles ? slot.roles.includes(c.member.role) : true);
    let pick = -1;
    const window = Math.min(remaining.length, config.slateWindow);
    for (let i = 0; i < window; i++) {
      if (wanted(remaining[i]!) && withinCaps(remaining[i]!, position)) {
        pick = i;
        break;
      }
    }
    if (pick < 0) pick = remaining.findIndex((c) => withinCaps(c, position));
    if (pick < 0) pick = 0;
    const [chosen] = remaining.splice(pick, 1);
    if (chosen!.tier === "headliner") headliners += 1;
    if (chosen!.member.followers > config.megaFollowers) megas += 1;
    ordered.push(chosen!);
  }
  return ordered.concat(remaining);
}
