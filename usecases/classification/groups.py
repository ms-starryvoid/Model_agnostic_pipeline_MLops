from contracts.usecase_groups import UsecaseGroupContract, InputSchema, OutputSchema


class TitanicClassificationGroup(UsecaseGroupContract):

    @property
    def usecase_name(self) -> str:
        return "shallow_mlp"

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            fields={
                "pclass":    int,
                "sex":       str,   # "male" | "female"
                "age":       float,
                "sibsp":     int,
                "parch":     int,
                "fare":      float,
                "embarked":  str,   # "S" | "C" | "Q"
            },
            description="One passenger record from the Titanic dataset.",
            example={
                "pclass": 3, "sex": "male", "age": 22.0,
                "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S",
            },
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            fields={"survived": bool, "probability": float},
            description="Survival prediction and model confidence.",
        )