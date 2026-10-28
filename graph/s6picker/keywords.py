"""S1's sensitive-class keyword list for S6 (preregistration §10, plan S6 / Vf29, the Y3 classes).

A card's member list is hidden when any of its top list names matches a term here,
case-insensitively, as a whole word or hashtag. The terms are copied from the frozen rules
file (docs/execution-plans/2026-10-28-whole-network-preregistration.md §10); "(s)"-style
suffixes are expanded. "any nationality or demonym used as an identity" is implemented as the
DEMONYMS list below (an explicit, reviewable stand-in; never inferred from a list's language).
"student(s) (with school-age cues)" matches only when the same name also carries a school-age cue.
"""
import re

CLASSES = {
    "news": ["news", "journalist", "journalists", "journalism", "reporter", "reporters", "media", "press",
             "breaking", "noticias", "nachrichten", "ニュース"],
    "politics": ["politic", "politics", "political", "election", "elections", "vote", "voting", "democrat",
                 "democrats", "republican", "republicans", "gop", "maga", "resist", "resistance", "progressive",
                 "conservative", "leftist", "liberal", "liberals", "socialist", "labour", "labor", "tory", "senate",
                 "congress", "parliament", "government", "activism", "activist", "activists", "antifascist", "antifa",
                 "政治", "política", "politik"],
    "religion": ["christian", "christians", "church", "catholic", "jesus", "bible", "faith", "muslim", "islam",
                 "jewish", "judaism", "torah", "hindu", "buddhist", "pagan", "witch", "witches", "witchcraft",
                 "atheist", "religion", "religious", "宗教"],
    "sexuality_gender": ["lgbt", "lgbtq", "lgbtqia", "queer", "gay", "lesbian", "bi", "bisexual", "trans",
                         "transgender", "nonbinary", "enby", "ace", "asexual", "pride", "2slgbtq"],
    "health": ["health", "mental health", "disabled", "disability", "chronic illness", "chronically ill", "autism",
               "autistic", "adhd", "long covid", "cancer", "spoonie", "recovery", "sober", "therapy"],
    "ethnicity_nationality": ["black", "african american", "latino", "latina", "latinx", "hispanic", "indigenous",
                              "native", "asian", "aapi", "jewish", "arab", "desi", "diaspora", "immigrant",
                              "immigrants"],
    "adult": ["nsfw", "18+", "adult", "porn", "onlyfans", "lewd", "kink", "fetish", "xxx", "nude", "nudes", "r18",
              "エロ"],
    "likely_minors": ["teen", "teens", "highschool", "high school", "middle school", "minor", "minors", "kids",
                      "13+", "14+", "15+", "16+", "17+", "mdni"],
}

# Nationalities and demonyms (ethnicity_nationality: "any nationality or demonym used as an identity").
DEMONYMS = """afghan albanian algerian american andorran angolan argentine argentinian armenian australian austrian
azerbaijani bahamian bahraini bangladeshi barbadian belarusian belgian belizean beninese bhutanese bolivian bosnian
botswanan brazilian brasileiro brasileira brasileiros brasileiras british bruneian bulgarian burkinabe burmese burundian
cambodian cameroonian canadian chadian chilean chinese colombian congolese costa rican croatian cuban cypriot czech
danish dominican dutch ecuadorian egyptian emirati english eritrean estonian ethiopian fijian filipino filipina finnish
french gabonese gambian georgian german ghanaian greek grenadian guatemalan guinean guyanese haitian honduran hungarian
icelandic indian indonesian iranian iraqi irish israeli italian ivorian jamaican japanese jordanian kazakh kenyan
korean kosovar kuwaiti kyrgyz lao laotian latvian lebanese liberian libyan lithuanian luxembourgish macedonian malagasy
malawian malaysian maldivian malian maltese mauritanian mauritian mexican moldovan mongolian montenegrin moroccan
mozambican namibian nepali nepalese nicaraguan nigerian nigerien norwegian omani pakistani palestinian panamanian
paraguayan persian peruvian polish portuguese puerto rican qatari romanian russian rwandan salvadoran samoan saudi
scottish senegalese serbian sierra leonean singaporean slovak slovenian somali south african spanish sri lankan sudanese
surinamese swazi swedish swiss syrian taiwanese tajik tanzanian thai togolese tongan trinidadian tunisian turkish turk
turkmen ugandan ukrainian uruguayan uzbek venezuelan vietnamese welsh yemeni zambian zimbabwean kurdish kurd catalan
basque tamil punjabi sikh roma romani maori sami tibetan uyghur chicano chicana boricua mexicano mexicana argentino
chileno chilena colombiano colombiana peruano peruana español española deutsche français française
italiano italiana türk 日本人 中国人 한국인""".split()
CLASSES["ethnicity_nationality"] = CLASSES["ethnicity_nationality"] + [
    "costa rican", "puerto rican", "sierra leonean", "south african", "sri lankan"] + [
    d for d in DEMONYMS if d not in ("costa", "rican", "puerto", "sierra", "leonean", "south", "african", "sri", "lankan")]

SCHOOL_AGE = re.compile(r"school|grade|class of|\bteen|\bkids?\b|year \d|学校|中学|高校|小学", re.I)
STUDENT = re.compile(r"(?<![\w])#?students?(?![\w])", re.I)


def _pattern(term):
    if re.search(r"[぀-ヿ一-鿿가-힯]", term):
        return re.escape(term)  # CJK and kana have no word boundaries: substring match.
    body = r"\s+".join(re.escape(w) for w in term.split())
    return r"(?<![\w])#?" + body + r"(?![\w+])"


PATTERNS = {cls: re.compile("|".join(_pattern(t) for t in terms), re.I) for cls, terms in CLASSES.items()}


def classes(name):
    """The Y3 classes a list name matches (sorted)."""
    hit = {cls for cls, p in PATTERNS.items() if p.search(name)}
    if STUDENT.search(name) and SCHOOL_AGE.search(name):
        hit.add("likely_minors")
    return sorted(hit)
