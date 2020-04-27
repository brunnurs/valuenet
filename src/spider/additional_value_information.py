class AdditionalValueInformation:
    def __init__(self, value_information) -> None:
        self.value_information_dic = {}

        for value, column, table in value_information:
            if value in self.value_information_dic:
                self.value_information_dic[value].append((value, column, table))
            else:
                self.value_information_dic[value] = [(value, column, table)]

    def enrich_value_by_additional_information(self, value):
        if value in self.value_information_dic:
            return self.value_information_dic[value]
        else:
            return [value]
