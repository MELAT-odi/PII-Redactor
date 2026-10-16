"""Generate synthetic Amharic PII records with split-aware vocabularies.

The generator writes a combined JSONL file containing train/validation/test
records, checks entity spans, and checks that person-name tokens do not cross
splits. All values are synthetic.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LEXICON_SEED = 2024          # FIXED. Changing it changes which names are in test.
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLITS = ("train", "validation", "test")
 
NEGATIVE_RATE = 0.25
FILLER_RATE = 0.35           # chance a positive gets harmless extra sentences
TITLE_RATE = 0.45
 
ENTITY_LABELS = {
    "PERSON_NAME", "PHONE", "NATIONAL_ID", "TIN", "BANK_ACCOUNT",
    "EMAIL", "ADDRESS_PLACE", "DATE_OF_BIRTH",
}
 
 
def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() or (parent / "README.md").exists():
            return parent
    return Path.cwd()
 
 
# ---------------------------------------------------------------------------
# Word pools (invented combinations only; no real person is described)
# ---------------------------------------------------------------------------
def uniq(items):
    return list(dict.fromkeys(items))
 
 
GIVEN_MALE = uniq("""
ዳዊት አቤል ዮሐንስ ሚካኤል ናትናኤል ታምራት ብርሃኑ አማኑኤል ዮሴፍ መስፍን ሳሙኤል ሄኖክ ቶማስ
ሰለሞን ኤፍሬም ኃይሌ ዮናስ ብሩክ ቴዎድሮስ ሚሊዮን ዘሪሁን ሳምሶን በቃሉ ምስጋናው ፍቃዱ አሸናፊ
ዓለም ደረጀ ፋሲል ኪሮስ ዳንኤል እዮብ ኤርምያስ ሀብታሙ ሰይፉ አብርሃም ሙሴ ኤልያስ ዋሲሁን ንጋቱ
መሐመድ አህመድ ኢብራሂም ጀማል ሐሰን ሁሴን ዑመር ከድር ገመቹ ቡልቻ ቶላ ጉዲና ዲሪባ ፈይሳ
ሌንጮ ደበላ ዋቆ ተሾመ በላይ ሲሳይ ተፈራ ዳግም ፍስሃ እንዳለ ብሩ ሰላማዊ
""".split())
 
GIVEN_FEMALE = uniq("""
ማርታ ሳራ ሀና ሰላም ቤተል የሺ ህይወት ሩት ሊዲያ ሄለን ኤልሳ ማህሌት ሰላማዊት አስቴር ኤደን
ናርዶስ ምህረት አልማዝ መሠረት ትዕግስት ፀሐይ ብርቱካን ሜሮን ቅድስት ሩቅያ ፍቅርተ ትርሲት
ሠናይት እየሩስ ዙፋን ፋጡማ አይሻ ሀሊማ ጽጌሬዳ ዓይናለም መቅደስ ሕይወት ሃና ቢሩክታዊት
ወይንሸት ሸዋዬ ፍቅርተ አበባ ሀብታም ብዙነሽ ሙሉ ዘነበች
""".split())
 
FATHER_NAMES = uniq("""
ተስፋዬ ደምሴ ከበደ ገብረ አለሙ መኮንን ታደሰ ሙሉጌታ ጌታቸው አየለ በቀለ ወልደ ኃይሉ ተክለ ግርማ
ዘለቀ አሰፋ ወርቁ ታምሩ አበበ ማሞ ወንድሙ ንጉሴ ጥላሁን ደስታ ሰይፉ አስፋው ቶላ ገመቹ ቡልቻ
ሁሴን አህመድ ዑመር መሐመድ በላይ ብርሃኑ ዓለሙ ገብሬ ዘውዴ እሸቱ ፈቃዱ ካሣ ሞገስ ኃይለማርያም
ገብረማርያም ወልደማርያም ገብረሚካኤል ወልደሚካኤል ተክሌ አምሳሉ ጸጋዬ ሽፈራው ደበበ ፍቅሬ
ዳኘው ጋሻው እንዳለ ቢያድግልኝ ሲሳይ ይርጋ ማሩ ሀይሌ ተፈራ ወንዴ ጌታነህ አባተ አሊ ጀማል
ክንፉ ዮሴፍ ፍስሃ ገብሩ ሀብቴ ዳግም ሰለሞን ዮሐንስ ፈይሳ ጉዲና ደበላ ተሾመ ዲሪባ ገብረኪዳን
ወልደገብርኤል ተሰማ ወጋየሁ አባይ ጥሩነህ ንጉሤ በየነ ገዛኸኝ
""".split())
 
TITLES_MALE = ["አቶ", "ዶ/ር", "ኢንጂነር", "መምህር", "ፕሮፌሰር"]
TITLES_FEMALE = ["ወይዘሮ", "ወይዘሪት", "ዶ/ር", "ኢንጂነር", "መምህርት"]
 
ADDIS_AREAS = uniq([w.replace("_", " ") for w in """
ቦሌ ገርጂ ሳሪስ ሰሚት ላፍቶ ጉለሌ ኮልፌ መርካቶ ፒያሳ ካዛንቺስ ቄራ ጎፋ ጀሞ ሚክሲኮ አያት ለቡ
ቃሊቲ ዊንጌት ሾላ ፉሪ ሰንጋ_ተራ አባኮራን ኮተቤ ሽሮ_ሜዳ እንጦጦ ቤቴል ሃያ_ሁለት ሲኤምሲ
""".split()])
 
SUB_CITIES = [
    "አዲስ ከተማ", "አቃቂ ቃሊቲ", "አራዳ", "ቦሌ", "ጉለሌ", "ልደታ", "ኮልፌ ቀራኒዮ",
    "ቂርቆስ", "የካ", "ንፋስ ስልክ ላፍቶ", "ለሚ ኩራ",
]
 
REGIONAL_CITIES = [w.replace("_", " ") for w in """
ባህር_ዳር ሀዋሳ መቀሌ ድሬዳዋ ጅማ ጎንደር አዳማ ደሴ ሐረር ደብረ_ማርቆስ ደብረ_ብርሃን አርባ_ምንጭ
ሻሸመኔ ነቀምቴ ጂንካ ሶዶ ሆሳዕና ሰመራ አሶሳ ጋምቤላ ወልቂጤ
""".split()]
 
STREETS = [
    "ቦሌ መንገድ", "ካዛንቺስ መንገድ", "አፍሪካ ጎዳና", "ቸርችል ጎዳና", "መካኒሳ መንገድ",
    "ሀያ ሁለት መንገድ", "ጃንሜዳ መንገድ", "ሳር ቤት መንገድ", "ደብረ ዘይት መንገድ", "ቤተል መንገድ",
    "ጉርድ ሾላ መንገድ", "ሃያ አራት መንገድ", "አምባሳደር መንገድ", "ሜክሲኮ መንገድ",
    "ጎጃም በር መንገድ", "ሰንጋተራ መንገድ",
]
ALWAYS_CITY = "አዲስ አበባ"   # present in every split on purpose (it is everywhere)
 
LATIN_FIRST = ["abel", "hana", "dawit", "martha", "samuel", "selam", "helen", "yonas",
               "ruth", "nathan", "betty", "meron", "kidus", "liya", "henok", "bethel",
               "eden", "biruk", "saron", "yohannes"]
LATIN_LAST = ["tesfaye", "kebede", "alemu", "bekele", "girma", "tadesse", "haile",
              "mekonnen", "abebe", "desta", "assefa", "negash", "worku", "tilahun"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "example.com",
                 "example.org"]
 
EC_MONTHS = ["መስከረም", "ጥቅምት", "ኅዳር", "ታኅሣሥ", "ጥር", "የካቲት", "መጋቢት", "ሚያዝያ",
             "ግንቦት", "ሰኔ", "ሐምሌ", "ነሐሴ"]
EC_MONTH_VARIANTS = {"ኅዳር": ["ኅዳር", "ህዳር"], "ታኅሣሥ": ["ታኅሣሥ", "ታህሳስ"]}
 
CUES = {
    "phone": ["ስልክ", "ስልክ ቁጥር", "ሞባይል", "የሞባይል ቁጥር", "ለመደወል", "Phone", "Tel"],
    "id": ["የፋይዳ ቁጥር", "የፋይዳ መታወቂያ ቁጥር", "የብሔራዊ መታወቂያ ቁጥር", "የመታወቂያ ቁጥር",
           "FIN", "National ID"],
    "tin": ["TIN ቁጥር", "TIN", "የግብር ከፋይ መለያ ቁጥር", "የግብር መለያ ቁጥር"],
    "bank": ["የባንክ ሂሳብ ቁጥር", "የሂሳብ ቁጥር", "አካውንት ቁጥር", "የንግድ ባንክ ሂሳብ ቁጥር",
             "Account No."],
    "email": ["ኢሜይል", "ኢሜል", "የኢሜይል አድራሻ", "Email"],
    "dob": ["የትውልድ ቀን", "የተወለደበት ቀን", "Date of birth", "DOB"],
}
 
 
# ---------------------------------------------------------------------------
# Deterministic partition of every pool into train / validation / test
# ---------------------------------------------------------------------------
def partition(items, ratios=SPLIT_RATIOS, seed=LEXICON_SEED):
    items = sorted(set(items))
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = n - n_val - n_test
    return {
        "train": items[:n_train],
        "validation": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
 
 
class Lexicon:
    """The words one split is allowed to use."""
 
    def __init__(self, split, names, areas, subs, cities, streets):
        self.split = split
        part = set(names[split])
        # Keep original order so output does not depend on set ordering.
        self.given_m = [n for n in GIVEN_MALE if n in part]
        self.given_f = [n for n in GIVEN_FEMALE if n in part]
        self.fathers = [n for n in FATHER_NAMES if n in part]
        self.areas = sorted(areas[split])
        self.subs = sorted(subs[split])
        self.regional = sorted(cities[split])
        self.streets = sorted(streets[split])
        for label, pool, minimum in [("given_m", self.given_m, 3), ("given_f", self.given_f, 3),
                                     ("fathers", self.fathers, 5), ("areas", self.areas, 3),
                                     ("subs", self.subs, 1), ("regional", self.regional, 2),
                                     ("streets", self.streets, 2)]:
            if len(pool) < minimum:
                raise RuntimeError(
                    f"Split '{split}' has only {len(pool)} entries for {label}. "
                    f"Add more words to the pools or change LEXICON_SEED."
                )
 
 
def build_lexicons():
    names = partition(uniq(GIVEN_MALE + GIVEN_FEMALE + FATHER_NAMES))
    areas = partition(ADDIS_AREAS)
    subs = partition(SUB_CITIES)
    cities = partition(REGIONAL_CITIES)
    streets = partition(STREETS)
    return {s: Lexicon(s, names, areas, subs, cities, streets) for s in SPLITS}, names
 
 
# ---------------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------------
GEEZ_ONES = "፩፪፫፬፭፮፯፰፱"
GEEZ_TENS = "፲፳፴፵፶፷፸፹፺"
 
 
def geez_number(n):
    """1..9999 in Ge'ez numerals, e.g. 1982 -> ፲፱፻፹፪. (Ge'ez has no zero, so it
    is used for dates and quantities, never for phone or ID numbers.)"""
    def two(x):
        t, o = divmod(x, 10)
        return (GEEZ_TENS[t - 1] if t else "") + (GEEZ_ONES[o - 1] if o else "")
    if n < 100:
        return two(n)
    h, r = divmod(n, 100)
    head = "" if h == 1 else two(h)
    return head + "፻" + (two(r) if r else "")
 
 
def digits(rng, n):
    return "".join(rng.choices("0123456789", k=n))
 
 
def group(s, sizes, sep):
    out, i = [], 0
    for k in sizes:
        out.append(s[i:i + k])
        i += k
    return sep.join(out)
 
 
# ---------------------------------------------------------------------------
# Entity generators. Each returns the exact string that becomes the span.
# ---------------------------------------------------------------------------
def gen_person(rng, lex, ctx):
    female = ctx["gender"] == "F"
    given = rng.choice(lex.given_f if female else lex.given_m)
    r = rng.random()
    if r < 0.12:
        return given
    father = rng.choice(lex.fathers)
    if r < 0.62:
        return f"{given} {father}"
    grandfather = rng.choice(lex.fathers)
    return f"{given} {father} {grandfather}"
 
 
def gen_phone(rng, lex, ctx):
    if rng.random() < 0.12:                       # Addis Ababa landline: 011 + 7 digits
        s = digits(rng, 7)
        return rng.choice([
            "011" + s, f"011 {s[:3]} {s[3:]}", f"011-{s[:3]}-{s[3:]}",
            f"+251 11 {s[:3]} {s[3:]}", f"+25111{s}",
        ])
    s = rng.choice("97") + digits(rng, 8)         # mobile: 09xxxxxxxx / 07xxxxxxxx
    forms = [
        ("0" + s, 36), ("+251" + s, 20),
        (f"+251 {s[:3]} {s[3:6]} {s[6:]}", 8),
        (f"0{s[:3]} {s[3:6]} {s[6:]}", 8),
        (f"0{s[:3]}-{s[3:6]}-{s[6:]}", 8),
        (f"0{s[:3]} {s[3:5]} {s[5:7]} {s[7:]}", 6),
        (f"+251-{s[:2]}-{s[2:5]}-{s[5:]}", 4),
        ("00251" + s, 4),
    ]
    return rng.choices([f for f, _ in forms], weights=[w for _, w in forms])[0]
 
 
def gen_national_id(rng, lex, ctx):
    s = digits(rng, 12)                           # Fayda FIN is 12 digits
    return rng.choices([s, group(s, [4, 4, 4], " "), group(s, [4, 4, 4], "-")],
                       weights=[70, 20, 10])[0]  # grouping is an ASSUMPTION
 
 
def gen_tin(rng, lex, ctx):
    return digits(rng, 10)                        # ASSUMPTION: 10 digits, verify officially
 
 
def gen_bank_account(rng, lex, ctx):
    if rng.random() < 0.6:
        return "1000" + digits(rng, 9)            # 13 digits, commercial-bank style
    return digits(rng, rng.choice([10, 12, 13, 14, 15, 16]))   # ASSUMPTION: varies by bank
 
 
def gen_email(rng, lex, ctx):
    a, b = rng.choice(LATIN_FIRST), rng.choice(LATIN_LAST)
    n = digits(rng, rng.choice([2, 3, 4]))
    local = rng.choice([f"{a}.{b}{n}", f"{a}{b}{n}", f"{a}_{b}{n}", f"{a}.{b[0]}{n}",
                        f"{b}.{a}{n}", f"{a}{n}"])
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"
 
 
def _date_parts(rng, year_lo, year_hi):
    return rng.randint(year_lo, year_hi), rng.randint(1, 12), rng.randint(1, 28)
 
 
def _ec_month(rng, idx):
    name = EC_MONTHS[idx]
    return rng.choice(EC_MONTH_VARIANTS.get(name, [name]))
 
 
def make_date(rng, birth):
    y, m, d = _date_parts(rng, 1950, 2012) if birth else _date_parts(rng, 2022, 2026)
    kinds = ["slash", "dash", "iso", "am", "geez"] if birth else ["slash", "dash", "iso", "am"]
    weights = [40, 15, 5, 30, 10] if birth else [45, 20, 10, 25]
    kind = rng.choices(kinds, weights=weights)[0]
    if kind == "slash":
        return f"{d:02d}/{m:02d}/{y}"
    if kind == "dash":
        return f"{d:02d}-{m:02d}-{y}"
    if kind == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    ec_year = y - 8                                # approximate Ethiopian-calendar year
    month = _ec_month(rng, m - 1)
    if kind == "am":
        return f"{d} {month} {ec_year}"
    return f"{geez_number(d)} {month} {geez_number(ec_year)}"
 
 
def gen_dob(rng, lex, ctx):
    return make_date(rng, birth=True)
 
 
def gen_address(rng, lex, ctx):
    form = rng.choices(range(6), weights=[3, 3, 3, 2, 2, 1])[0]
    kebele = f"{rng.randint(1, 25):02d}"
    woreda = f"{rng.randint(1, 15):02d}"
    house = rng.randint(1, 999)
    if form == 0:
        return f"{rng.choice(lex.areas)}{rng.choice(['፣', '፣ '])}{rng.choice(lex.streets)}"
    if form == 1:
        return f"{rng.choice(lex.subs)} ክፍለ ከተማ ወረዳ {woreda} ቀበሌ {kebele} የቤት ቁጥር {house}"
    if form == 2:
        return f"{ALWAYS_CITY}፣ {rng.choice(lex.areas)}"
    if form == 3:
        return (f"{rng.choice(lex.areas)}፣ {rng.choice(lex.streets)}፣ "
                f"የቤት ቁጥር {house}")
    if form == 4:
        return f"{rng.choice(lex.regional)}፣ ቀበሌ {kebele}"
    return f"{ALWAYS_CITY}፣ {rng.choice(lex.subs)} ክፍለ ከተማ"
 
 
GENERATORS = {
    "PERSON_NAME": gen_person, "PHONE": gen_phone, "NATIONAL_ID": gen_national_id,
    "TIN": gen_tin, "BANK_ACCOUNT": gen_bank_account, "EMAIL": gen_email,
    "ADDRESS_PLACE": gen_address, "DATE_OF_BIRTH": gen_dob,
}
 
# ---------------------------------------------------------------------------
# Sentence frames.  {LABEL} = an annotated entity.  {cue:x}, {T}, {digits:N},
# {num:a-b}, {date} = plain (unlabelled) text.
# Test frames are never used for training or validation.
# ---------------------------------------------------------------------------
TRAIN_FRAMES = [
    "የአመልካቹ ስም {PERSON_NAME} ነው። ስልኩ {PHONE} ነው።",
    "የደንበኛው ስም {PERSON_NAME} ሲሆን {cue:id} {NATIONAL_ID} ነው።",
    "የግብር ከፋዩ {PERSON_NAME} {cue:tin} {TIN} ነው።",
    "የደንበኛው {PERSON_NAME} {cue:bank} {BANK_ACCOUNT} ነው።",
    "የአመልካቹ {PERSON_NAME} {cue:email} {EMAIL} ነው።",
    "አመልካቹ {PERSON_NAME} የሚኖረው {ADDRESS_PLACE} ነው።",
    "የአመልካቹ ሙሉ ስም {PERSON_NAME} {cue:dob} {DATE_OF_BIRTH} ነው።",
    "የአመልካቹ ስም {PERSON_NAME}፣ {cue:phone} {PHONE}፣ {cue:email} {EMAIL}፣ አድራሻ {ADDRESS_PLACE} ነው።",
    "ስም: {PERSON_NAME}\nስልክ: {PHONE}\nአድራሻ: {ADDRESS_PLACE}",
    "Name: {PERSON_NAME}\nPhone: {PHONE}\nEmail: {EMAIL}",
    "ውድ {T}{PERSON_NAME}፣\nጥያቄዎ ደርሶናል። ለተጨማሪ መረጃ በስልክ ቁጥር {PHONE} ይደውሉ።",
    "{T}{PERSON_NAME} የተወለዱት {DATE_OF_BIRTH} ሲሆን የመኖሪያ አድራሻቸው {ADDRESS_PLACE} ነው።",
    "የሂሳብ ባለቤት: {PERSON_NAME}፣ የሂሳብ ቁጥር: {BANK_ACCOUNT}፣ ስልክ: {PHONE}",
    "የማመልከቻ ቅጽ\nስም: {PERSON_NAME}\nየትውልድ ቀን: {DATE_OF_BIRTH}\n{cue:id}: {NATIONAL_ID}\nስልክ: {PHONE}",
    "{PERSON_NAME}, {PHONE}, {ADDRESS_PLACE}",
    "{PERSON_NAME},{EMAIL},{PHONE}",
    "ሰላም፣ እኔ {PERSON_NAME} ነኝ። ስልክ ቁጥሬ {PHONE} ነው።",
    "ደብዳቤው የተጻፈው ለሚከተለው ሰው ነው፦ {T}{PERSON_NAME}",
    "የተመዝጋቢው {PERSON_NAME} የግብር መለያ ቁጥር {TIN}፣ የባንክ ሂሳብ {BANK_ACCOUNT} ነው።",
    "የታካሚው ስም {PERSON_NAME}፣ ዕድሜ {num:1-90} ዓመት፣ ስልክ {PHONE}",
    "አቤቱታ አቅራቢ: {T}{PERSON_NAME}\nአድራሻ: {ADDRESS_PLACE}\nየፋይዳ ቁጥር: {NATIONAL_ID}",
    "እባክዎ {T}{PERSON_NAME} ን ያነጋግሩ፤ ኢሜይል {EMAIL}።",
    "ስለ {T}{PERSON_NAME} የተሰጠ መረጃ፦ የትውልድ ቀን {DATE_OF_BIRTH}፣ ስልክ {PHONE}።",
    "የተማሪው ስም {PERSON_NAME}፣ የወላጅ ስልክ {PHONE}፣ የመኖሪያ አድራሻ {ADDRESS_PLACE}።",
    "{T}{PERSON_NAME}\n{cue:phone}: {PHONE}\n{cue:email}: {EMAIL}",
    "የደንበኛ ምዝገባ፦ {PERSON_NAME} | {PHONE} | {EMAIL}",
    "ገንዘቡ ወደ {T}{PERSON_NAME} {cue:bank} {BANK_ACCOUNT} ተላልፏል።",
    "የተጠቃሚው ስም {PERSON_NAME} ሲሆን በ{num:2010-2018} ዓ.ም ተመዝግቧል። ስልክ {PHONE}።",
    "Hello {PERSON_NAME}, please confirm your phone {PHONE} and your address {ADDRESS_PLACE}.",
]
 
TEST_FRAMES = [
    "የክፍያ ማረጋገጫ\nከፋይ: {PERSON_NAME}\nየተከፈለበት ሂሳብ: {BANK_ACCOUNT}\nመጠን: {num:100-50000} ብር\nየአድራሻ ስልክ: {PHONE}",
    "{T}{PERSON_NAME} ({PHONE}) ስብሰባውን እንዲያስተባብሩ ተመድበዋል።",
    "የታካሚ ምዝገባ ቅጽ\nየታካሚው ስም {PERSON_NAME}\nየትውልድ ቀን {DATE_OF_BIRTH}\nየመኖሪያ አድራሻ {ADDRESS_PLACE}\nየቤተሰብ ስልክ {PHONE}",
    "id,name,phone,email\n{num:1-999},{PERSON_NAME},{PHONE},{EMAIL}",
    "ከላይ የተጠቀሰው ተማሪ {PERSON_NAME} ሲሆን የወላጅ ስልክ ቁጥር {PHONE} ነው።",
    "{PERSON_NAME}\n{ADDRESS_PLACE}\n{PHONE}",
    "የባንክ ዝርዝር: ባለቤት {PERSON_NAME}፣ አካውንት {BANK_ACCOUNT}፣ TIN {TIN}",
    "Dear {T}{PERSON_NAME},\nYour registration is confirmed. Please contact us at {EMAIL}.",
    "የፋይዳ ቁጥር {NATIONAL_ID} ያለው {T}{PERSON_NAME} ተመዝግበዋል።",
    "{PERSON_NAME}፤ TIN {TIN}፤ ስልክ {PHONE}",
    "ቀጠሮ ለያዘው {T}{PERSON_NAME} ማሳሰቢያ ተልኳል። የተወለዱበት ቀን {DATE_OF_BIRTH} ተብሎ ተመዝግቧል።",
]
 
# Neutral sentences added around positives. They contain no PII.
FILLER = [
    "ከዚህ በታች ያለውን መረጃ ይመልከቱ።", "ማመልከቻው ተመዝግቧል።", "ጥያቄዎ በሂደት ላይ ነው።",
    "እባክዎ መረጃውን በጥንቃቄ ያስቀምጡ።", "ለትብብርዎ እናመሰግናለን።", "ጉዳዩ በቅርቡ ይታያል።",
    "ሰነዱ ለሚመለከተው አካል ተልኳል።", "ይህ መረጃ ሚስጥራዊ ነው።", "ቀጠሮው ለሚቀጥለው ሳምንት ተይዟል።",
    "ማንኛውም ጥያቄ ካለዎት ያሳውቁን።", "ውሳኔው በጽሑፍ ይላካል።",
]
 
# Negatives. These must contain NO annotated PII.
NEGATIVE_SENTENCES = [
    # weather / education / work / health / business
    "ዛሬ አየሩ ጥሩ ነው።", "ከሰዓት በኋላ ዝናብ ሊዘንብ ይችላል።", "በጠዋት አየሩ ቀዝቃዛ ነበር።",
    "በዚህ ሳምንት ፀሐያማ ቀናት ይኖራሉ።", "ነገ ደመናማ ይሆናል ተብሎ ይጠበቃል።",
    "ተማሪዎቹ በክፍል ውስጥ ትምህርት እየተማሩ ነው።", "መምህሩ አዲስ ምዕራፍ አስተማሩ።",
    "የቤተ መጻሕፍቱ ክፍል ዛሬ ክፍት ነው።", "ቡድኑ ለፈተናው በጋራ ተዘጋጀ።",
    "የትምህርት ቤቱ ግቢ በጣም ሰፊ ነው።", "ሰራተኞቹ የሳምንቱን ሥራ ጨርሰዋል።",
    "የቡድኑ ስብሰባ በጠዋት ተካሄደ።", "አዲሱ ፕሮጀክት በጥሩ ሁኔታ እየሄደ ነው።",
    "ሰነዱ ከመጨረሻው ግምገማ በኋላ ይላካል።", "ሪፖርቱ እስከ ሳምንቱ መጨረሻ ይጠናቀቃል።",
    "ሆስፒታሉ ከጠዋቱ እስከ ማታ ክፍት ነው።", "እባክዎ በቂ ውሃ ይጠጡ።", "ክትባቱ ለሁሉም በነጻ ይሰጣል።",
    "ሱቁ በየቀኑ ከጠዋቱ {num:1-5} ሰዓት ጀምሮ ይከፈታል።", "የኩባንያው ትርፍ በዚህ ዓመት ጨምሯል።",
    "ደንበኞች ጥራት ያለው አገልግሎት ይፈልጋሉ።",
    # public places in event sentences (not a person's address)
    "ስብሰባው በአዲስ አበባ ዩኒቨርሲቲ ተካሄደ።", "የቦሌ ዓለም አቀፍ አውሮፕላን ማረፊያ በጣም ሥራ የበዛበት ነው።",
    "ትርኢቱ በመስቀል አደባባይ ይካሄዳል።", "ባቡሩ ከመርካቶ ወደ ገርጂ ይሄዳል።",
    "ኮንፈረንሱ በባህር ዳር ተካሂዷል።",
    # ordinary words that are also names
    "ሰላም ለሁሉም ይሁን።", "ሰላም የሁሉም ፍላጎት ነው።", "ህይወት ቀላል አይደለም።", "ፍቅር ታላቅ ኃይል ነው።",
    "ተስፋ አትቁረጥ።", "ብርሃን ከጨለማ ይሻላል።", "አበባው በጣም ያማረ ነው።",
    "ትዕግስት የስኬት መሠረት ነው።", "መሠረት ያለው ትምህርት ያስፈልጋል።", "ፀሐይ በጠዋት ትወጣለች።",
    "ምህረት ማድረግ መልካም ነው።", "ዓለም በፍጥነት እየተቀየረች ነው።", "ብርቱካን በገበያ ላይ ርካሽ ነው።",
    "አልማዝ ውድ ድንጋይ ነው።", "ሰላማዊ ሰልፍ ተካሄደ።", "ሀብታም ሰው ሁሉ ደስተኛ አይደለም።",
    # numbers that are NOT personal identifiers
    "የደረሰኝ ቁጥር {digits:10} ነው።", "የትዕዛዝ ቁጥር {digits:8} ተመዝግቧል።",
    "የፋይል ማጣቀሻ ቁጥር {digits:6}/{num:2010-2018} ነው።", "የጭነቱ መከታተያ ቁጥር {digits:12} ነው።",
    "የምርቱ ሞዴል ቁጥር {digits:13} ነው።", "የሰነዱ መለያ ቁጥር {digits:10} ነው።",
    "ዋጋው {num:50-9000} ብር ነው።", "ጠቅላላ ክፍያ {num:1000-99999} ብር ነው።",
    "የመጽሐፉ ምዕራፍ ቁጥር {num:1-30} ነው።", "ከተማዋ {num:100000-5000000} ነዋሪዎች አሏት።",
    "ቡድኑ {num:5-40} አባላት አሉት።", "ስብሰባው ቀን {date} ይካሄዳል።", "የክፍያ ቀነ ገደብ {date} ነው።",
    "ሪፖርቱ {date} ተልኳል።",
    # cue words with no value (forms, instructions, warnings)
    "ስልክ ቁጥርዎን ያስገቡ።", "እባክዎ ኢሜይልዎን ይጻፉ።", "የባንክ ሂሳብ ቁጥርዎን ከታች ይሙሉ።",
    "የትውልድ ቀንዎን ያስገቡ።", "የፋይዳ ቁጥርዎን ለማንም አይጋሩ።", "የይለፍ ቃልዎን ለማንም አይስጡ።",
    "ስም: ________\nስልክ: ________\nአድራሻ: ________", "የግብር ከፋይ መለያ ቁጥር: ________",
]
 
# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SLOT = re.compile(r"\{([^{}]+)\}")
 
 
def plain_slot(slot, rng, ctx):
    if slot == "T":
        if rng.random() < TITLE_RATE:
            return rng.choice(TITLES_FEMALE if ctx["gender"] == "F" else TITLES_MALE) + " "
        return ""
    if slot == "date":
        return make_date(rng, birth=False)
    kind, _, arg = slot.partition(":")
    if kind == "cue":
        return rng.choice(CUES[arg])
    if kind == "digits":
        return digits(rng, int(arg))
    if kind == "num":
        lo, hi = arg.split("-")
        return str(rng.randint(int(lo), int(hi)))
    raise ValueError(f"Unknown slot: {{{slot}}}")
 
 
def render(frame, rng, lex, ctx, text=""):
    """Fill a frame. Returns (text, entities). `text` is a prefix already written."""
    entities, pos = [], 0
    for m in SLOT.finditer(frame):
        text += frame[pos:m.start()]
        pos = m.end()
        slot = m.group(1)
        if slot in ENTITY_LABELS:
            value = GENERATORS[slot](rng, lex, ctx)
            start = len(text)
            text += value
            entities.append({"start": start, "end": len(text), "label": slot, "text": value})
        else:
            text += plain_slot(slot, rng, ctx)
    text += frame[pos:]
    return text, entities
 
 
def make_positive(rng, lex, split, neg_pool):
    frames = TEST_FRAMES if split == "test" else TRAIN_FRAMES
    ctx = {"gender": rng.choice("MF")}
    prefix = ""
    if rng.random() < FILLER_RATE and rng.random() < 0.6:
        prefix = rng.choice(FILLER) + rng.choice([" ", "\n"])
    text, entities = render(rng.choice(frames), rng, lex, ctx, text=prefix)
    if rng.random() < FILLER_RATE and not prefix:
        text += rng.choice([" ", "\n"]) + rng.choice(FILLER)
    return text, entities
 
 
def make_negative(rng, lex, split, neg_pool):
    ctx = {"gender": rng.choice("MF")}
    parts = rng.sample(neg_pool, k=rng.choice([1, 1, 2, 3]))
    text = ""
    for i, sentence in enumerate(parts):
        if i:
            text += rng.choice([" ", "\n"])
        text, _ = render(sentence, rng, lex, ctx, text=text)
    return text, []
 
 
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_record(record):
    text, rid = record["text"], record["id"]
    previous_end = 0
    for e in sorted(record["entities"], key=lambda x: x["start"]):
        if e["label"] not in ENTITY_LABELS:
            raise ValueError(f"{rid}: unknown label {e['label']}")
        if not (0 <= e["start"] < e["end"] <= len(text)):
            raise ValueError(f"{rid}: bad boundaries {e}")
        if text[e["start"]:e["end"]] != e["text"]:
            raise ValueError(f"{rid}: span text mismatch {e}")
        if e["text"] != e["text"].strip():
            raise ValueError(f"{rid}: span has leading/trailing whitespace {e}")
        if e["start"] < previous_end:
            raise ValueError(f"{rid}: overlapping entities {e}")
        previous_end = e["end"]
 
 
def check_leakage(records):
    """Fail loudly if any name token appears in more than one split."""
    tokens = defaultdict(set)
    for r in records:
        for e in r["entities"]:
            if e["label"] == "PERSON_NAME":
                tokens[r["split"]].update(e["text"].split())
    problems = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        shared = tokens[a] & tokens[b]
        if shared:
            problems.append(f"{a} and {b} share name tokens: {sorted(shared)[:10]}")
    if problems:
        raise RuntimeError("NAME LEAKAGE: " + "; ".join(problems))
    return {s: len(t) for s, t in tokens.items()}
 
 
# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate(counts, seed, out_file, lexicon_dir):
    lexicons, name_parts = build_lexicons()
    neg_parts = partition(NEGATIVE_SENTENCES)
    records, seen = [], defaultdict(set)
 
    for split in SPLITS:
        rng = random.Random(f"{seed}-{split}")
        lex, neg_pool = lexicons[split], neg_parts[split]
        n = counts[split]
        negatives = set(rng.sample(range(n), round(n * NEGATIVE_RATE)))
        for i in range(n):
            maker = make_negative if i in negatives else make_positive
            for _ in range(30):                      # avoid exact duplicate texts
                text, entities = maker(rng, lex, split, neg_pool)
                if text not in seen[split]:
                    break
            else:
                raise RuntimeError(
                    f"Could not generate a unique record for {split} item {i + 1}. "
                    "Increase the split count's vocabulary or reduce the requested count."
                )
            seen[split].add(text)
            record = {"id": f"synthetic_{split}_{i + 1:05d}", "split": split,
                      "text": text, "entities": entities}
            validate_record(record)
            records.append(record)
 
    name_tokens = check_leakage(records)
 
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
 
    lexicon_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (lexicon_dir / f"names_{split}.txt").write_text(
            "\n".join(sorted(name_parts[split])) + "\n", encoding="utf-8")
        places = lexicons[split].areas + lexicons[split].subs + lexicons[split].regional \
            + lexicons[split].streets
        (lexicon_dir / f"places_{split}.txt").write_text(
            "\n".join(sorted(places)) + "\n", encoding="utf-8")
 
    # Report
    print(f"Saved {len(records)} records to {out_file}")
    for split in SPLITS:
        rs = [r for r in records if r["split"] == split]
        neg = sum(1 for r in rs if not r["entities"])
        labels = Counter(e["label"] for r in rs for e in r["entities"])
        print(f"\n[{split}] records={len(rs)} negatives={neg} "
              f"({neg / len(rs):.0%}) distinct name tokens={name_tokens.get(split, 0)}")
        print("   " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())))
    print("\nLeakage check passed: no name token is shared between splits.")
    print(f"Lexicon files written to {lexicon_dir}")
 
 
def main():
    root = find_project_root()
    ap = argparse.ArgumentParser(description="Generate synthetic Amharic PII data with leakage-aware splits.")
    ap.add_argument("--train", type=int, default=3200)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=root / "data" / "synthetic" / "raw_annotated.jsonl")
    ap.add_argument("--lexicon-dir", type=Path, default=root / "data" / "lexicons")
    a = ap.parse_args()
    counts = {"train": a.train, "validation": a.val, "test": a.test}
    if any(count <= 0 for count in counts.values()):
        ap.error("--train, --val, and --test must all be positive integers")
    generate(counts, a.seed, a.out, a.lexicon_dir)
 
 
if __name__ == "__main__":
    main()