def get_prompts_race():
    properties = {
        "num_white": {
            "type": "string",
            "description": "The number of white/caucasian patients in the study",
        },
        "num_black": {
            "type": "string",
            "description": "The number of black/african american patients in the study",
        },
        "num_latino": {
            "type": "string",
            "description": "The number of latino patients in the study",
        },
        "num_asian": {
            "type": "string",
            "description": "The number of asian patients in the study",
        },
        "evidence_span_race": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_white, num_black, num_latino, and num_asian.",
        },
    }

    functions = [
        {
            "name": "extract_patient_nums_by_race",
            "description": "Get the number of patients in this study for different gender and race demographics.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "num_white",
                    "num_black",
                    # "num_latino",
                    # "num_asian",
                    "evidence_span_race",
                ],
            },
        },
    ]

    content_str = """### QUESTION: How many patients were in the study, broken down by race?

###  STUDY: {input}"""
    return properties, functions, content_str


def get_prompts_gender():
    properties = {
        "num_male": {
            "type": "string",
            "description": "The number of male patients in the study",
        },
        "num_female": {
            "type": "string",
            "description": "The number of female patients in the study",
        },
        "num_total": {
            "type": "string",
            "description": "The total number of patients in the study",
        },
        "evidence_span_gender": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_male, num_female, and num_gender.",
        },
    }

    functions = [
        {
            "name": "extract_patient_nums_by_gender",
            "description": "Get the number of patients in this study for different genders",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "num_male",
                    "num_female",
                    "num_total",
                    "evidence_span_gender",
                ],
            },
        },
    ]

    content_str = """### QUESTION: How many male and female patients were in the study?

###  STUDY: {input}"""
    return properties, functions, content_str


def get_prompts_gender_and_race():
    properties = {
        "num_white": {
            "type": "string",
            "description": "The number of white/caucasian patients in the study",
        },
        "num_black": {
            "type": "string",
            "description": "The number of black/african american patients in the study",
        },
        "num_latino": {
            "type": "string",
            "description": "The number of latino patients in the study",
        },
        "num_asian": {
            "type": "string",
            "description": "The number of asian patients in the study",
        },
        "evidence_span_race": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_white, num_black, num_latino, and num_asian.",
        },
        "num_male": {
            "type": "string",
            "description": "The number of male patients in the study",
        },
        "num_female": {
            "type": "string",
            "description": "The number of female patients in the study",
        },
        "num_total": {
            "type": "string",
            "description": "The total number of patients in the study",
        },
        "evidence_span_gender": {
            "type": "string",
            "description": "The long text span in the input that includes evidence for num_male, num_female, and num_gender.",
        },
    }

    functions = [
        {
            "name": "extract_patient_nums_by_demographics",
            "description": "Get the number of patients in this study for different gender and race demographics.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "num_male",
                    "num_female",
                    "num_total",
                    "evidence_span_gender",
                ],
            },
        },
    ]

    content_str = """### QUESTION: How many patients were in the study, broken down by race and gender?

###  STUDY: {input}"""
    return properties, functions, content_str
