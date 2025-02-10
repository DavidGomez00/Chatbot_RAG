from apiclient import discovery
from httplib2 import Http
import os
import logging
import json
from oauth2client import client, file, tools

SCOPES = "https://www.googleapis.com/auth/drive"
DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"


def authenticate():
    '''Authenticate with Google API'''
    store = file.Storage(".google_auth/token.json")
    creds = store.get()

    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets(".google_auth/client_secret_906762387182-a2fhr2j368mvj0okq09felitnfpp9fi8.apps.googleusercontent.com.json", SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build(
        "forms",
        "v1",
        http=creds.authorize(Http()),
        discoveryServiceUrl=DISCOVERY_DOC,
        static_discovery=False,
    )
    return form_service

def create_form(form_service, form_number, model, embedding, chunk_str):
    ### Form creation ###
    form = {
        "info": {
            "documentTitle": f"Evaluación de resultados Chatbot VIHv3 - {form_number}.",
            "title": "Evaluación de resultados Chatbot VIH",
        }
    }
    create_result = form_service.forms().create(body=form).execute()
    formId = create_result["formId"]

    ### Form update ###
    requests = []
    ##### Form description #####
    with open(".data/form_description_template.txt", "r") as f:
        template = f.read()
    description = template.format(
        model_name=model.split('/')[2],
        embedding=embedding,
        chunk_type='Secciones naturales' if chunk_str == 'nat' else 'División semántica'
    )
    updateFormInfo = {
        "updateFormInfo": {
            "info": {
                "description": description,
            },
            "updateMask": "description",
        }
    }
    requests.append(updateFormInfo)

    ##### Placeholder items #####
    for _ in range(26):
        item, index = _create_item(query_path=None, placeholder=True)
        createItem = {
            "createItem": 
            {
                "item": item,
                "location": {
                    "index": index,
                }
            }
        }
        requests.append(createItem)
    update = {"requests": requests}
    update_form(update, form_service, formId)
    #print("Form updated.")
    return formId

def update_form(update, form_service, formId):
    '''Update an existing form'''
    # Update the form with a description
    question_setting = (
        form_service.forms()
        .batchUpdate(formId=formId, body=update)
        .execute()
    )
    return question_setting

def _create_item(query_path, placeholder=False):
    '''Create an item for the form from a query file or a placeholder
    with index 0.'''
    if placeholder:
        index = 0
        title = "Title placeholder"
        description = "Description placeholder"
    else:
        index = int(query_path.split("_")[1].split(".")[0]) 
        query = json.load(open(query_path, "r"))
        title = query["query"][:-1] # Remove the last character "\n"
        description = query["answer"]
        # Remove all * from the answer
        # description = description.split("</think>")[1] # Deepseek specific
        description = description.replace("*", "")

    item = {
                "title": title,   
                "description": description,
                "questionItem": {
                    "question": {
                        "required": True,
                        "scaleQuestion": {
                            "low": 1,
                            "high": 5,
                            "lowLabel": "Deficiente",
                            "highLabel": "Excelente"
                        }
                    }
                }
            }
    return item, index

def create_queries_update(queries_path):
    '''Create a request to update a form'''
    requests = []
    
    for query in os.listdir(queries_path):
        print(os.path.join(model, embedding, chunk_str, query))
        item, index = _create_item(query_path=os.path.join(model, embedding, chunk_str, query))
        
        updateItem = {
            "updateItem": {
                "item": item,
                "location": {
                    "index": index
                },
                "updateMask": "title,description"
            }
        }
        requests.append(updateItem)
    
    update = {
        "requests": requests
    }
    
    return update


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    form_service = authenticate()
    models = ["results/nemotron/"]
    form_number = 1
    for model in models:
        for embedding in os.listdir(model):
            for chunk_str in os.listdir(os.path.join(model, embedding)):      
                logging.info(f"Creating form for {model.split('/')[-2]}, {embedding}, {chunk_str}...")
                formId = create_form(form_service, form_number, model, embedding, chunk_str)
                form_number += 1
                queries_path = os.path.join(model, embedding, chunk_str)
                update = create_queries_update(queries_path)
                update_form(update, form_service, formId=formId)

    logging.info("Forms created and updated.")