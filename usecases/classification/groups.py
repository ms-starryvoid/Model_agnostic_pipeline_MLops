from contracts.usecase_groups import UsecaseGroupContract, InputSchema, OutputSchema


class TitanicClassificationGroup(UsecaseGroupContract):

    @property
    def usecase_name(self) -> str:
        return "shallow_mlp"

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            fields={
                "Pclass":    int,
                "Sex":       str,   # "male" | "female"
                "Age":       float,
                "Sibsp":     int,
                "Parch":     int,
                "Fare":      float,
                "Embarked":  str,   # "S" | "C" | "Q"
            },
            description="One passenger record from the Titanic dataset.",
            example={
                "Pclass": 3, "Sex": "male", "Age": 22.0,
                "Sibsp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S",
            },
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            fields={"survived": bool, "probability": float},
            description="Survival prediction and model confidence.",
        )