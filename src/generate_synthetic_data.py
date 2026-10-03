import json
import random
from pathlib import Path

#-----------------------
# Project Paths
#-----------------------

project_root = Path(__file__).resolve().parent.parent

output_file = (
    project_root / "data"/"synthetic"/"raw_annotated.jsonl"
)

# ---------------------------------------------------------
# FAKE DATA
# ---------------------------------------------------------

first_names =[
    "ሚሊዮን",
    "ማርታ",
    "ሳራ",
    "አቤል",
    "ሀና",
    "ዳዊት",
    "ሰላም",
    "ቤተል",
    "ዮሐንስ",
    "ሚካኤል",
]

father_names = [
    "ተስፋዬ",
    "ደምሴ",
    "ከበደ",
    "ገብረ",
    "አለሙ",
    "መኮንን",
    "ታደሰ",
    "ሙሉጌታ",
]

places =[
    "ቦሌ",
    "የካ",
    "አራዳ",
    "ቂርቆስ",
    "ልደታ",
    "አዲስ አበባ",
    "መርካቶ",
    "ፒያሳ",
]

streets = [
    "ቦሌ መንገድ",
    "ካዛንቺስ መንገድ",
    "አፍሪካ አቬኑ",
    "ቸርችል ጎዳና",
]


# ---------------------------------------------------------
# FAKE PII GENERATORS
# ---------------------------------------------------------

def generate_person():
    first = random.choice(first_names)
    father =random.choice(father_names)

    return f"{first} {father}"

def generate_phone():
    prefix = random.choice([
        "09",
        "07",
        "+2519",
        "+2517"
    ])

    remaining ="".join(
        random.choices("0123456789", k=8)
    )

    return prefix +remaining

def generate_national_id():
    # Synthetic 12-digit identifier
    return "".join(random.choices("0123456789", k=12))

def generate_tin():
     # Synthetic TIN-like nuber
     return "".join(random.choices("0123456789", k=10))

def generate_bank_account():
    # Synthetic bank-account 
    return "".join(random.choices("0123456789", k=13))

def generate_email():
    number = random.randint(10000, 99999)

    return f"person{number}@example.com"

def generate_date_of_birth():
    year = random.randint(1970, 2005)
    month = random.randint(1,12)
    day = random.randint(1,28)

    return f"{day:02d}/{month:02d}/{year}"


def generate_address():
    place = random.choice(places)
    street = random.choice(streets)

    return f"{place}፣{street}"

# ---------------------------------------------------------
# ENTITY HELPER
# ---------------------------------------------------------

def add_entity(text,value,label):
    """
    Add a piece of PII to text and record its character span.
    """

    start = len(text)
    text += value
    end=len(text)

    entity={
        "start": start,
        "end": end,
        "label":label,
        "text":value,
    }

    return text, entity

# ---------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------

def template_person_phone():
    person =generate_person()
    phone =generate_phone()

    text ="የአመልካቹ ስም "

    text, person_entity =add_entity(
        text,
        person,
        "PERSON_NAME" 
    )

    text += " ነው። ስልኩ "

    text, phone_entity = add_entity(
        text,
        phone,
        "PHONE"
    )

    text += " ነው።"

    return text, [person_entity, phone_entity]


def template_id_name():
    person = generate_person()
    national_id = generate_national_id()

    text = "የደንበኛው ስም "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " ሲሆን የመታወቂያ ቁጥሩ "

    text, id_entity = add_entity(
        text,
        national_id,
        "NATIONAL_ID"
    )

    text += " ነው።"

    return text, [person_entity, id_entity]


def template_tin():
    person = generate_person()
    tin = generate_tin()

    text = "የግብር ከፋዩ "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " TIN ቁጥር "

    text, tin_entity = add_entity(
        text,
        tin,
        "TIN"
    )

    text += " ነው።"

    return text, [person_entity, tin_entity]


def template_bank():
    person = generate_person()
    account = generate_bank_account()

    text = "የደንበኛው "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " የባንክ ሂሳብ ቁጥር "

    text, account_entity = add_entity(
        text,
        account,
        "BANK_ACCOUNT"
    )

    text += " ነው።"

    return text, [person_entity, account_entity]


def template_email():
    person = generate_person()
    email = generate_email()

    text = "የአመልካቹ "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " ኢሜይል "

    text, email_entity = add_entity(
        text,
        email,
        "EMAIL"
    )

    text += " ነው።"

    return text, [person_entity, email_entity]


def template_address():
    person = generate_person()
    address = generate_address()

    text = "አመልካቹ "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " የሚኖረው "

    text, address_entity = add_entity(
        text,
        address,
        "ADDRESS_PLACE"
    )

    text += " ነው።"

    return text, [person_entity, address_entity]


def template_dob():
    person = generate_person()
    dob = generate_date_of_birth()

    text = "የ"

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += " የትውልድ ቀን "

    text, dob_entity = add_entity(
        text,
        dob,
        "DATE_OF_BIRTH"
    )

    text += " ነው።"

    return text, [person_entity, dob_entity]


def template_multiple():
    person = generate_person()
    phone = generate_phone()
    email = generate_email()
    address = generate_address()

    text = "የአመልካቹ ስም "

    text, person_entity = add_entity(
        text,
        person,
        "PERSON_NAME"
    )

    text += "፣ ስልክ "

    text, phone_entity = add_entity(
        text,
        phone,
        "PHONE"
    )

    text += "፣ ኢሜይል "

    text, email_entity = add_entity(
        text,
        email,
        "EMAIL"
    )

    text += "፣ አድራሻ "

    text, address_entity = add_entity(
        text,
        address,
        "ADDRESS_PLACE"
    )

    text += " ነው።"

    return text, [
        person_entity,
        phone_entity,
        email_entity,
        address_entity,
    ]


# ---------------------------------------------------------
# DATASET GENERATION
# ---------------------------------------------------------

TEMPLATES = [
    template_person_phone,
    template_id_name,
    template_tin,
    template_bank,
    template_email,
    template_address,
    template_dob,
    template_multiple,
]


def generate_dataset(number_of_records=500):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for i in range(number_of_records):

            template = random.choice(TEMPLATES)

            text, entities = template()

            record = {
                "id": f"synthetic_{i + 1:05d}",
                "text": text,
                "entities": entities,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Generated {number_of_records} records."
    )

    print(
        f"Saved to: {output_file}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    generate_dataset(500)