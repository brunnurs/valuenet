import requests

import json
import requests


def remote_named_entity_recognition(document, ner_api_secret):

    assert ner_api_secret and ner_api_secret != 'PLEASE_ADD_YOUR_OWN_GOOGLE_API_KEY_HERE', "Please add you Google API Key for Named Entity Recognition"

    payload = {
        "document": {
            "type": "PLAIN_TEXT",
            "content": document
        },
        "encodingType": 'UTF8'
    }

    parameters = {
        'key': ner_api_secret
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
