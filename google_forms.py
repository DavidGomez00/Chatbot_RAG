from apiclient import discovery
from httplib2 import Http
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

def create_form(form_service):
    form = {
        "info": {
            "title": "My new form",
        },
    }
    # Prints the details of the sample form
    result = form_service.forms().create(body=form).execute()
    print(result)


if __name__ == "__main__":
    form_service = authenticate()
    create_form(form_service)