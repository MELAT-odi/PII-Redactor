import json
import re
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
output_file = (
    project_root / "data" / "synthetic" / "external_eval.jsonl"
)

token_pattern = re.compile(
    r"\S+|[።፣፤፥፦፧፨,.;:!?()\[\]{}]"
)

# Each sentence is written as separate segments.
# A segment contains text and either a PII category or None.
# All values are invented for evaluation purposes.

examples = [
    [
        ("ለቢሮው", None),
        ("ማህሌት ገብረ", "PERSON_NAME"),
        ("በ", None),
        ("0912345678", "PHONE"),
        ("ማግኘት", None),
        ("ይቻላል", None),
    ],
    [
        ("የአመልካቹ", None),
        ("ስም", None),
        ("ዳዊት ከበደ", "PERSON_NAME"),
        ("ነው", None),
        ("TIN", None),
        ("1234567890", "TIN"),
    ],
    [
        ("ደብዳቤውን", None),
        ("ወደ", None),
        ("abebe.fake@example.com", "EMAIL"),
        ("መላክ", None),
        ("ይቻላል", None),
    ],
    [
        ("አዲሱ", None),
        ("አድራሻ", None),
        ("ቦሌ መድኃኒዓለም", "ADDRESS_PLACE"),
        ("አካባቢ", None),
        ("ነው", None),
    ],
    [
        ("የተመዘገበው", None),
        ("የባንክ", None),
        ("ሂሳብ", None),
        ("1000123456789", "BANK_ACCOUNT"),
        ("ነው", None),
    ],
    [
        ("የደንበኛው", None),
        ("መለያ", None),
        ("ET-FAKE-583920", "NATIONAL_ID"),
        ("በፋይሉ", None),
        ("ውስጥ", None),
        ("ተመዝግቧል", None),
    ],
    [
        ("የትውልድ", None),
        ("ቀኑ", None),
        ("12/05/1998", "DATE_OF_BIRTH"),
        ("መሆኑ", None),
        ("ተገልጿል", None),
    ],
    [
        ("አቶ", None),
        ("ናትናኤል ተስፋዬ", "PERSON_NAME"),
        ("በ", None),
        ("ቂርቆስ ክፍለ ከተማ", "ADDRESS_PLACE"),
        ("ይኖራሉ", None),
    ],
    [
        ("ለተጨማሪ", None),
        ("መረጃ", None),
        ("0923456781", "PHONE"),
        ("ወይም", None),
        ("contact.fake@example.org", "EMAIL"),
        ("ይጠቀሙ", None),
    ],
    [
        ("የአበባ", None),
        ("ሙሉ", "PERSON_NAME"),
        ("TIN", None),
        ("9876543210", "TIN"),
        ("በማመልከቻው", None),
        ("ላይ", None),
        ("ተጽፏል", None),
    ],
    [
        ("የተቀባዩ", None),
        ("ስም", None),
        ("ሰላማዊት ደረጀ", "PERSON_NAME"),
        ("እና", None),
        ("የባንክ", None),
        ("ቁጥር", None),
        ("1000987654321", "BANK_ACCOUNT"),
        ("ተረጋግጠዋል", None),
    ],
    [
        ("የመኖሪያ", None),
        ("ቦታው", None),
        ("አያት መንገድ", "ADDRESS_PLACE"),
        ("ሲሆን", None),
        ("ስልኩ", None),
        ("0934567821", "PHONE"),
        ("ነው", None),
    ],
    [
        ("የፋይል", None),
        ("ባለቤት", None),
        ("ብርሃኑ አለሙ", "PERSON_NAME"),
        ("የተወለዱት", None),
        ("በ", None),
        ("03-11-2001", "DATE_OF_BIRTH"),
        ("ነው", None),
    ],
    [
        ("እባክዎ", None),
        ("የተቋሙን", None),
        ("ደብዳቤ", None),
        ("ወደ", None),
        ("melat.test@example.net", "EMAIL"),
        ("ይላኩ", None),
    ],
    [
        ("በ", None),
        ("ልደታ ክፍለ ከተማ", "ADDRESS_PLACE"),
        ("የሚገኘው", None),
        ("ቢሮ", None),
        ("የ", None),
        ("ፋይል", None),
        ("FAKE-ID-729104", "NATIONAL_ID"),
        ("ተቀብሏል", None),
    ],
    [
        ("የግብር", None),
        ("ከፋይ", None),
        ("ሚካኤል ጌታቸው", "PERSON_NAME"),
        ("መለያ", None),
        ("4567891230", "TIN"),
        ("ነው", None),
    ],
    [
        ("የአገልግሎቱ", None),
        ("ተጠቃሚ", None),
        ("ፍሬወይኒ ኃይሉ", "PERSON_NAME"),
        ("በ", None),
        ("ሲኤምሲ አካባቢ", "ADDRESS_PLACE"),
        ("ይገኛሉ", None),
    ],
    [
        ("የክፍያ", None),
        ("ማረጋገጫው", None),
        ("የሂሳብ", None),
        ("ቁጥር", None),
        ("1000456789123", "BANK_ACCOUNT"),
        ("እና", None),
        ("0911122233", "PHONE"),
        ("ይዟል", None),
    ],
    [
        ("ለ", None),
        ("ህጋዊ", None),
        ("ግንኙነት", None),
        ("የሚያገለግለው", None),
        ("legal.fake@example.com", "EMAIL"),
        ("ነው", None),
    ],
    [
        ("የተመዝጋቢው", None),
        ("ማስረጃ", None),
        ("NID-FAKE-628401", "NATIONAL_ID"),
        ("እና", None),
        ("የልደት", None),
        ("ቀን", None),
        ("21/09/1995", "DATE_OF_BIRTH"),
        ("ተካትተዋል", None),
    ],
]


def build_record(example, index):
    tokens = []
    labels = []

    for segment_text, category in example:
        segment_tokens = token_pattern.findall(segment_text)

        for token_index, token in enumerate(segment_tokens):
            tokens.append(token)

            if category is None:
                labels.append("O")
            elif token_index == 0:
                labels.append(f"B-{category}")
            else:
                labels.append(f"I-{category}")

    if len(tokens) != len(labels):
        raise ValueError(f"Token/label mismatch in example {index}")

    return {
        "id": f"external_eval_{index:03d}",
        "tokens": tokens,
        "labels": labels,
    }


def main():
    records = [
        build_record(example, index)
        for index, example in enumerate(examples, start=1)
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("External evaluation dataset created.")
    print(f"Documents: {len(records)}")
    print(f"Output: {output_file}")

    category_counts = {}

    for record in records:
        for label in record["labels"]:
            if label.startswith("B-"):
                category = label[2:]
                category_counts[category] = (
                    category_counts.get(category, 0) + 1
                )

    print("\nEntity counts:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
