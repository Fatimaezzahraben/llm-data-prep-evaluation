"""
app/services/agent_c_dataset_hints.py
=====================================
OPTIONAL per-dataset supplements for Agent C's LLM semantic check
(app/services/agent_c.py::validate_semantics).

Agent C's core design is intentionally dataset-AGNOSTIC (see the docstring at the
top of agent_c.py) -- everything is auto-discovered from the data so a brand-new
dataset works with zero added code. This file does NOT change that: auto-discovery
stays the default and the fallback for every dataset, including ones not listed
here. All this module adds is optional `semantic_hints` (plain-English description
of what a column means) and `valid_values_map` (the fixed vocabulary of a closed
categorical column) for the datasets already present under datasets/ -- both are
parameters agent_c.validate_semantics() / validate() already accept, this just
fills them in per dataset instead of leaving the LLM to guess.

Only genuinely closed/fixed vocabularies are listed in valid_values_map (codes like
Sex, Embarked, meal type -- not open-ended real-world entities like hospital/city
names, which get a semantic_hint instead so the LLM still uses judgment rather than
being locked to a fixed list).
"""

DATASET_HINTS = {
    # -------------------------------------------------------------------
    # flights (7 cols) -- classic multi-source flight-status reconciliation
    # benchmark: several websites (`src`) report possibly-conflicting times
    # for the same `flight`. No metadata.json for this one (not generated
    # via the project's own noise injector), so hints are based on the raw
    # column contents themselves.
    # -------------------------------------------------------------------
    "flights": {
        "semantic_hints": {
            "src": (
                "the name/code of the website or system that reported this flight "
                "status row (airline sites, airport codes, or news/travel "
                "aggregators, e.g. 'aa', 'ord', 'flightaware', 'orbitz'). Many "
                "different legitimate sources exist here by design -- this is NOT "
                "a single category to normalize. Only flag a value if it looks "
                "like corrupted/garbled text, never merely because it is an "
                "unfamiliar source name."
            ),
            "flight": (
                "a flight identifier in the format '<AIRLINE>-<FLIGHT NUMBER>-"
                "<ORIGIN AIRPORT CODE>-<DEST AIRPORT CODE>' (e.g. "
                "'AA-3859-IAH-ORD'). Flag it only if the structure itself is "
                "broken (missing a segment, garbled characters) -- the specific "
                "airline/number/airport combination is not something to judge."
            ),
            "sched_dep_time": (
                "a scheduled departure time of day in 'H:MM a.m.'/'H:MM p.m.' "
                "format (e.g. '7:10 a.m.', '11:45 p.m.'), hour 1-12 and minute "
                "00-59, or a missing-value marker. Flag only a genuinely broken "
                "time (impossible hour/minute, garbled text), not a merely "
                "unusual-but-valid time of day."
            ),
            "act_dep_time": (
                "an actual departure time of day, same 'H:MM a.m./p.m.' format "
                "as sched_dep_time -- naturally differs from the scheduled time "
                "(that is the point of this column), do not flag a difference "
                "from sched_dep_time as an error, only a broken/garbled format."
            ),
            "sched_arr_time": (
                "a scheduled arrival time of day, same 'H:MM a.m./p.m.' format "
                "as sched_dep_time."
            ),
            "act_arr_time": (
                "an actual arrival time of day, same 'H:MM a.m./p.m.' format as "
                "sched_dep_time -- naturally differs from the scheduled arrival "
                "time, do not flag that difference, only a broken/garbled format."
            ),
        },
        "valid_values_map": {},
    },

    # -------------------------------------------------------------------
    # hospital (20 cols) -- CMS hospital-quality benchmark limited to a
    # fixed, closed universe of 45 real hospitals in Alabama + Alaska
    # (the classic HoloClean/BigDansing "Hospital" dataset). No
    # metadata.json, but a real run's DIRTY evaluation (OpenRouter, see
    # conversation) confirmed corruption hits HospitalName ("name"),
    # Address1, City, State, ProviderNumber, ZipCode and PhoneNumber via
    # character-substitution typos (e.g. 'o'->'x': "hxspital").
    # -------------------------------------------------------------------
    "hospital": {
        "semantic_hints": {
            "ProviderNumber": (
                "a 5-digit CMS hospital provider ID number, digits only, no "
                "letters -- a value with a stray letter/placeholder character "
                "(e.g. 'x') in place of a digit is a real error, not a variant."
            ),
            "HospitalName": (
                "the legal name of a real hospital, one of a fixed set of 45 "
                "hospitals located in Alabama or Alaska."
            ),
            "Address1": (
                "a US street address for one of the 45 hospitals in this "
                "dataset (Alabama or Alaska) -- e.g. '1720 university blvd'."
            ),
            "City": "a US city name, in Alabama or Alaska.",
            "ZipCode": "a 5-digit US ZIP code, digits only, no letters.",
            "CountyName": "a US county name, in Alabama or Alaska.",
            "PhoneNumber": (
                "a 10-digit US phone number, digits only, no formatting "
                "characters (no dashes/parentheses/spaces expected)."
            ),
            "MeasureName": (
                "the full official English description of a CMS hospital "
                "quality measure (a long sentence), one of a fixed set of 28."
            ),
            "Score": (
                "a quality-measure score, either a percentage like '97%' "
                "(0%-100%) or the literal string 'empty' when not reported -- "
                "not a free-text field."
            ),
            "Sample": (
                "the sample size behind a Score, formatted as '<N> patients' "
                "(e.g. '33 patients', '0 patients') or the literal string "
                "'empty' when not reported."
            ),
        },
        "valid_values_map": {
            "State": ["ak", "al"],
            "HospitalOwner": [
                "government - federal",
                "government - hospital district or authority",
                "government - local",
                "government - state",
                "proprietary",
                "voluntary non-profit - church",
                "voluntary non-profit - other",
                "voluntary non-profit - private",
            ],
            "EmergencyService": ["no", "yes"],
            "Condition": [
                "children s asthma care",
                "heart attack",
                "heart failure",
                "pneumonia",
                "surgical infection prevention",
            ],
            "MeasureCode": [
                "ami-1", "ami-2", "ami-3", "ami-4", "ami-5", "ami-7a", "ami-8a",
                "cac-1", "cac-2", "cac-3",
                "hf-1", "hf-2", "hf-3", "hf-4",
                "pn-2", "pn-3b", "pn-4", "pn-5c", "pn-6", "pn-7",
                "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3",
                "scip-inf-4", "scip-inf-6", "scip-vte-1", "scip-vte-2",
            ],
            "Stateavg": [
                "ak_ami-1", "ak_ami-2", "ak_hf-1", "ak_hf-2", "ak_hf-3", "ak_hf-4",
                "ak_pn-2", "ak_pn-3b", "ak_pn-4", "ak_pn-5c", "ak_pn-6", "ak_pn-7",
                "ak_scip-card-2", "ak_scip-inf-1", "ak_scip-inf-2", "ak_scip-inf-3",
                "ak_scip-inf-4", "ak_scip-inf-6", "ak_scip-vte-1", "ak_scip-vte-2",
                "al_ami-1", "al_ami-2", "al_ami-3", "al_ami-4", "al_ami-5",
                "al_ami-7a", "al_ami-8a", "al_cac-1", "al_cac-2", "al_cac-3",
                "al_hf-1", "al_hf-2", "al_hf-3", "al_hf-4", "al_pn-2", "al_pn-3b",
                "al_pn-4", "al_pn-5c", "al_pn-6", "al_pn-7", "al_scip-card-2",
                "al_scip-inf-1", "al_scip-inf-2", "al_scip-inf-3", "al_scip-inf-4",
                "al_scip-inf-6", "al_scip-vte-1", "al_scip-vte-2",
            ],
            "HospitalName": [
                "alaska regional hospital", "andalusia regional hospital",
                "baptist medical center south", "callahan eye foundation hospital",
                "cherokee medical center", "chilton medical center",
                "community hospital inc", "coosa valley medical center",
                "crenshaw community hospital", "cullman regional medical center",
                "dale medical center", "decatur general hospital",
                "dekalb regional medical center",
                "east alabama medical center and snf", "elba general hospital",
                "eliza coffee memorial hospital", "fayette medical center",
                "flowers hospital", "g h lanier memorial hospital",
                "gadsden regional medical center", "georgiana hospital",
                "hartselle medical center", "helen keller memorial hospital",
                "huntsville hospital", "jackson hospital & clinic inc",
                "marion regional medical center", "marshall medical center north",
                "marshall medical center south", "medical center enterprise",
                "mizell memorial hospital", "northwest medical center",
                "prattville baptist hospital", "riverview regional medical center",
                "russellville hospital", "shelby baptist medical center",
                "southeast alabama medical center",
                "southwest alabama medical center", "st vincents blount",
                "st vincents east", "st vincents hospital",
                "stringfellow memorial hospital",
                "univ of south alabama medical center",
                "university of alabama hospital", "wedowee hospital",
                "yukon kuskokwim delta reg hospital",
            ],
            "Address1": [
                "1000 first street north", "1007 goodyear avenue",
                "101 hospital circle", "101 sivley rd", "1010 lay dam road",
                "1108 ross clark circle", "1201 7th street se",
                "124 s memorial dr", "1256 military street south",
                "126 hospital ave", "1300 south montgomery avenue",
                "150 gilbreath drive", "15155 highway 43",
                "1530 u s highway 43", "1653 temple avenue north",
                "1720 university blvd", "1725 pine street",
                "1912 alabama highway 157", "200 med center drive",
                "2000 pepperell parkway", "201 pine street northwest",
                "205 marengo street", "209 north main street",
                "2105 east south boulevard", "2451 fillingim street",
                "2505 u s highway 431 north", "2801 debarr road",
                "301 east 18th st", "315 w hickory st", "33700 highway 43",
                "400 n edwards street", "400 northwood dr",
                "4370 west main street", "4800 48th st",
                "50 medical park east drive", "515 miranda st",
                "600 south third street", "619 south 19th street",
                "702 n main st", "8000 alabama highway 69",
                "805 friendship road", "810 st vincents drive",
                "849 south three notch street", "987 drayton street",
                "po box 287",
            ],
            "City": [
                "alabaster", "anchorage", "andalusia", "anniston", "bethel",
                "birmingham", "boaz", "centre", "clanton", "cullman", "decatur",
                "dothan", "elba", "enterprise", "fayette", "florence",
                "fort payne", "gadsden", "georgiana", "guntersville", "hamilton",
                "hartselle", "huntsville", "luverne", "mobile", "montgomery",
                "oneonta", "opelika", "opp", "ozark", "prattville",
                "russellville", "sheffield", "sylacauga", "tallassee",
                "thomasville", "valley", "wedowee", "winfield",
            ],
            "CountyName": [
                "anchorage", "autauga", "bethel", "blount", "butler", "calhoun",
                "chambers", "cherokee", "chilton", "clarke", "coffee",
                "covington", "crenshaw", "cullman", "dale", "de kalb", "elmore",
                "etowah", "fayette", "franklin", "houston", "jefferson",
                "lauderdale", "lee", "madison", "marion", "marshall", "mobile",
                "montgomery", "morgan", "randolph", "shelby", "talladega",
            ],
            "MeasureName": [
                "all heart surgery patients whose blood sugar (blood glucose) is kept under good control in the days right after surgery",
                "children and their caregivers who received a home management plan of care document while hospitalized for asthma",
                "children who received reliever medication while hospitalized for asthma",
                "children who received systemic corticosteroid medication (oral and iv medication that reduces inflammation and controls symptoms) while hospitalized for asthma",
                "heart attack patients given ace inhibitor or arb for left ventricular systolic dysfunction (lvsd)",
                "heart attack patients given aspirin at arrival",
                "heart attack patients given aspirin at discharge",
                "heart attack patients given beta blocker at discharge",
                "heart attack patients given fibrinolytic medication within 30 minutes of arrival",
                "heart attack patients given pci within 90 minutes of arrival",
                "heart attack patients given smoking cessation advice/counseling",
                "heart failure patients given ace inhibitor or arb for left ventricular systolic dysfunction (lvsd)",
                "heart failure patients given an evaluation of left ventricular systolic (lvs) function",
                "heart failure patients given discharge instructions",
                "heart failure patients given smoking cessation advice/counseling",
                "patients who got treatment  at the right time (within 24 hours before or after their surgery) to help prevent blood clots after certain types of surgery",
                "pneumonia patients assessed and given influenza vaccination",
                "pneumonia patients assessed and given pneumococcal vaccination",
                "pneumonia patients given initial antibiotic(s) within 6 hours after arrival",
                "pneumonia patients given smoking cessation advice/counseling",
                "pneumonia patients given the most appropriate initial antibiotic(s)",
                "pneumonia patients whose initial emergency room blood culture was performed prior to the administration of the first hospital dose of antibiotics",
                "surgery patients needing hair removed from the surgical area before surgery who had hair removed using a safer method (electric clippers or hair removal cream c not a razor)",
                "surgery patients who were given an antibiotic at the right time (within one hour before surgery) to help prevent infection",
                "surgery patients who were given the  right kind  of antibiotic to help prevent infection",
                "surgery patients who were taking heart drugs called beta blockers before coming to the hospital who were kept on the beta blockers during the period just before and after their surgery",
                "surgery patients whose doctors ordered treatments to prevent blood clots after certain types of surgeries",
                "surgery patients whose preventive antibiotics were stopped at the right time (within 24 hours after surgery)",
            ],
        },
    },

    # -------------------------------------------------------------------
    # hotel-booking-demand (32 cols). metadata.json documents exactly which
    # columns get which error family injected per noise level: typos_text
    # on hotel/deposit_type/customer_type, typos_numeric on lead_time/
    # adults, missing on country/agent/children/market_segment/meal,
    # format errors on reservation_status_date.
    # -------------------------------------------------------------------
    "hotel-booking-demand": {
        "semantic_hints": {
            "country": "an ISO 3166-1 alpha-3 country code (e.g. 'PRT', 'GBR', 'USA').",
            "agent": "a numeric travel agent ID -- a plain integer-like number.",
            "company": "a numeric company ID -- a plain integer-like number.",
            "lead_time": (
                "the number of days between the booking date and the arrival "
                "date, a plain non-negative integer with no unit suffix or "
                "extra characters."
            ),
            "adults": (
                "the number of adult guests on the booking, a plain "
                "non-negative integer with no unit suffix or extra characters."
            ),
        },
        "valid_values_map": {
            "hotel": ["City Hotel", "Resort Hotel"],
            "arrival_date_month": [
                "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December",
            ],
            "meal": ["BB", "FB", "HB", "SC", "Undefined"],
            "market_segment": [
                "Aviation", "Complementary", "Corporate", "Direct", "Groups",
                "Offline TA/TO", "Online TA", "Undefined",
            ],
            "distribution_channel": ["Corporate", "Direct", "GDS", "TA/TO", "Undefined"],
            "reserved_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
            "assigned_room_type": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"],
            "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
            "customer_type": ["Contract", "Group", "Transient", "Transient-Party"],
            "reservation_status": ["Canceled", "Check-Out", "No-Show"],
        },
    },

    # -------------------------------------------------------------------
    # titanic (12 cols). metadata.json documents typos_text on Sex/Embarked
    # and typos_numeric on Fare/Age -- a real run already showed the exact
    # kind of corruption these produce on Fare ('8.4583B', '16.0kg', '26.0'
    # with a stray unit/character appended).
    # -------------------------------------------------------------------
    "titanic": {
        "semantic_hints": {
            "Sex": "the passenger's sex.",
            "Embarked": "the port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton.",
            "Cabin": (
                "a cabin identifier (a deck letter followed by a room number, e.g. "
                "'C85') -- often missing, and a wide variety of real values is "
                "expected, do not flag an unfamiliar-but-plausible code."
            ),
            "Fare": (
                "the ticket fare, a plain decimal number (e.g. '7.25', "
                "'71.2833') -- no currency symbol or unit suffix. A value with "
                "a trailing/embedded letter or symbol (e.g. '16.0kg', "
                "'8.4583B') is a real error, not a variant."
            ),
            "Age": (
                "the passenger's age in years, a plain non-negative number "
                "(can be fractional for infants, e.g. '0.42') -- no unit "
                "suffix or extra characters."
            ),
        },
        "valid_values_map": {
            "Sex": ["female", "male"],
            "Embarked": ["C", "Q", "S"],
        },
    },
}


def get_dataset_hints(dataset_name: str) -> dict:
    """
    Returns the optional {"semantic_hints": {...}, "valid_values_map": {...}}
    overrides for a known dataset name (case/hyphen/underscore-insensitive).

    For any dataset name not listed in DATASET_HINTS -- including a totally new
    dataset never seen before -- this returns empty dicts, so agent_c.validate()'s
    own auto-discovery handles everything on its own, exactly as if this module
    didn't exist.
    """
    if not dataset_name:
        return {"semantic_hints": {}, "valid_values_map": {}}
    key_norm = dataset_name.strip().lower().replace("_", "-")
    for name, hints in DATASET_HINTS.items():
        if name.lower().replace("_", "-") == key_norm:
            return {
                "semantic_hints": dict(hints.get("semantic_hints", {})),
                "valid_values_map": dict(hints.get("valid_values_map", {})),
            }
    return {"semantic_hints": {}, "valid_values_map": {}}
