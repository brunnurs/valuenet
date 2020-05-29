import requests

import json
import requests


def remote_named_entity_recognition(document):
    payload = {
        "document": {
            "type": "PLAIN_TEXT",
            "content": document
        },
        "encodingType": 'UTF8'
    }

    parameters = {
        'key': 'NOT VALID ANYMORE - PLEASE CREATE YOUR OWN ACCOUNT!'
    }

    url = 'https://language.googleapis.com/v1beta2/documents:analyzeEntities'

    response = requests.post(url, json=payload, params=parameters)

    if response.status_code != 200:
        print("ERROR!!! HTTP: {}. for request '{}'".format(response.status_code, document))
        print(response.text)
        return None
    else:
        print("HTTP: {}. for request '{}'".format(response.status_code, document))
        return json.loads(response.text)


if __name__ == "__main__":
    result = remote_named_entity_recognition("Find job id and date of hire for those employees who was hired between November 5th, 2007 and July 5th, 2009.")
    if result:
        print(result)
