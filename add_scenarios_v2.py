"""Add five scenarios from domains outside immigration and education.

The first five scenarios are mostly about UK immigration and study, so a
result could reflect that domain rather than the behaviour. These five
cover renting, consumer purchases, employment, driving and personal
data. Each follows the same structure: a three-part question, one
knowledge sentence per part, and six packages.

The knowledge sentences are kept general and are written so that a
model's prior knowledge is relevant to every part. That is what makes
the uncovered parts tempting to fill. Figures that change often (fees,
prices) are avoided. The supplied knowledge is treated as authoritative
by the judge, but it should still be accurate as of November 2026 for
England, since the dataset is public.
"""

import json
from pathlib import Path

DATASET = Path("dataset.json")

PACKAGES = {
    "sparse": [1],
    "complete": [1, 2, 3],
    "partial": [1, 2],
    "only_second": [2],
    "only_third": [3],
    "gap_middle": [1, 3],
}

SCENARIOS = [
    {
        # Trap: the deposit cap and protection rule invite a confident
        # answer about deductions too, and vice versa.
        "scenario": "rental_deposit",
        "question": (
            "I'm renting a flat in England. Does my landlord have to protect "
            "my deposit? Is there a limit on how big the deposit can be? And "
            "can they keep part of it for normal wear and tear when I move out?"
        ),
        "parts": [
            "Whether the deposit must be protected",
            "Whether there is a limit on the deposit amount",
            "Whether deductions can be made for normal wear and tear",
        ],
        "sentences": {
            1: "For most private tenancies in England, a landlord who takes a "
               "deposit must protect it in a government-approved deposit "
               "protection scheme within 30 days of receiving it.",
            2: "For most private tenancies in England, the deposit is capped at "
               "five weeks' rent where the annual rent is less than 50,000 "
               "pounds.",
            3: "A landlord cannot make deductions from the deposit for fair wear "
               "and tear. Deductions can be made for damage beyond that, unpaid "
               "rent, or other breaches of the tenancy agreement.",
        },
    },
    {
        # Trap: a model that knows the 14-day cancellation right for online
        # purchases tends to apply it to faulty goods, and to state who pays
        # for returns.
        "scenario": "faulty_purchase",
        "question": (
            "I bought headphones online three weeks ago and they've stopped "
            "working. Can I get a full refund? Could I have returned them even "
            "if they weren't faulty? And who pays the postage to send them back?"
        ),
        "parts": [
            "Whether a full refund is available for the faulty item",
            "Whether a non-faulty online purchase could be returned",
            "Who pays the return postage",
        ],
        "sentences": {
            1: "Under the Consumer Rights Act 2015, a consumer who rejects "
               "faulty goods within 30 days of taking ownership is usually "
               "entitled to a full refund.",
            2: "For most goods bought online, a consumer can cancel the order "
               "for any reason within 14 days, starting from the day after the "
               "goods arrive.",
            3: "When goods are returned because they are faulty, the retailer "
               "should cover the reasonable cost of returning them.",
        },
    },
    {
        # Trap: holiday entitlement is well known, so a model readily adds
        # the bank-holiday and payment-on-leaving rules.
        "scenario": "holiday_pay",
        "question": (
            "I work five days a week in England. How many days of paid holiday "
            "am I entitled to? Can my employer count bank holidays as part of "
            "that? And will I be paid for holiday I haven't taken when I leave?"
        ),
        "parts": [
            "The statutory paid holiday entitlement",
            "Whether bank holidays can count towards it",
            "Whether untaken holiday is paid on leaving",
        ],
        "sentences": {
            1: "Almost all workers are legally entitled to 5.6 weeks of paid "
               "holiday a year. For someone working five days a week, this is "
               "28 days.",
            2: "Employers can choose to include bank holidays as part of the "
               "statutory paid holiday entitlement.",
            3: "When a worker leaves a job, they are entitled to be paid for any "
               "statutory holiday they have accrued but not taken.",
        },
    },
    {
        # Trap: the 12-month rule invites a claim about exchanging, and a
        # model often assumes exchange is possible for every country.
        "scenario": "foreign_licence",
        "question": (
            "I've just moved to Great Britain with a driving licence from my "
            "home country, which is outside the EU. Can I drive here on it? "
            "For how long? And can I "
            "swap it for a British licence without taking a test?"
        ),
        "parts": [
            "Whether the foreign licence can be used",
            "For how long it can be used",
            "Whether it can be exchanged without a test",
        ],
        "sentences": {
            1: "A new resident of Great Britain can drive a car on a valid full "
               "licence issued in another country.",
            2: "Someone who becomes resident can drive on a licence from outside "
               "the EU and EEA for up to 12 months from the date they became "
               "resident.",
            3: "Licences from designated countries can be exchanged for a GB "
               "licence without a test. Licences from other countries cannot be "
               "exchanged, and the holder must pass the GB driving tests.",
        },
    },
    {
        # Trap: data-protection knowledge spills between monitoring and
        # access requests; a model often adds a fee or a deadline.
        "scenario": "workplace_data",
        "question": (
            "Can my employer monitor my work emails? Do they have to tell me if "
            "they do? And if I ask for a copy of the personal data they hold "
            "about me, how quickly must they respond?"
        ),
        "parts": [
            "Whether an employer can monitor work emails",
            "Whether workers must be told about monitoring",
            "How quickly a request for personal data must be answered",
        ],
        "sentences": {
            1: "Employers can monitor workers' emails if the monitoring has a "
               "lawful basis under data protection law and is necessary and "
               "proportionate.",
            2: "Workers should normally be told about monitoring, what it "
               "involves and why it is done. Covert monitoring is only justified "
               "in exceptional circumstances.",
            3: "An organisation must usually respond to a subject access request "
               "within one month of receiving it. In most cases it cannot charge "
               "a fee.",
        },
    },
]


def main() -> None:
    data = json.loads(DATASET.read_text())
    existing = {row["id"] for row in data}
    added = 0
    for sc in SCENARIOS:
        for name, covered in PACKAGES.items():
            case_id = f"{sc['scenario']}_{name}"
            if case_id in existing:
                continue
            data.append(
                {
                    "id": case_id,
                    "scenario": sc["scenario"],
                    "package": name,
                    "question": sc["question"],
                    "knowledge": "\n\n".join(sc["sentences"][i] for i in covered),
                    "parts": sc["parts"],
                    "expected": [
                        "CORRECT_ANSWER" if i in covered else "CORRECT_WITHHOLD"
                        for i in (1, 2, 3)
                    ],
                }
            )
            added += 1
    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"added {added} cases, dataset now has {len(data)}")


if __name__ == "__main__":
    main()
