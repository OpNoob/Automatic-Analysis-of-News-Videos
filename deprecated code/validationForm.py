import copy

# google-api-python-client
# google-auth
# google-auth-oauthlib
# google-auth-httplib2

from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# SCOPES = ['https://www.googleapis.com/auth/forms', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/forms.body', 'https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = "secret_data/client_secret.json"
FORM_NAME = "News Video Validation"
FORM_ID = '1DPxUuBzXOJ3I9UvC94oMIy_ODOaTCubPYDpuCKBGw80'
SECTION_NAME = "Validation"

flow = InstalledAppFlow.from_client_secrets_file(
    CREDENTIALS_FILE,
    scopes=SCOPES
)

# Run the authentication flow and get the user's credentials
credentials = flow.run_local_server(port=0)

# Create a service object for the Google Forms API
forms_service = build('forms', 'v1', credentials=credentials)

form = forms_service.forms().get(formId=FORM_ID).execute()
print(form)

remove_everything = False
remove_indexes = list()
last_item_index = 0
for index, item in enumerate(form["items"]):
    if remove_everything:
        remove_indexes.append(index)
    else:
        last_item_index = index

    # If section name found, delete everything after it
    if "title" in item:
        if item["title"] == SECTION_NAME:
            remove_everything = True

request_list = list()
if remove_indexes:
    for index in sorted(remove_indexes, reverse=True):
        removeRequest = {
            "deleteItem": {
                "location": {
                    "index": index
                }
            }
        }
        request_list.append(removeRequest)
else:
    if not remove_everything:  # Section not found, then create a new one
        last_item_index += 1

        createRequest = {
            "createItem": {
                "item": {
                    'pageBreakItem': {},
                    'title': SECTION_NAME,
                },
                "location": {
                    "index": last_item_index
                }
            }
        }
        request_list.append(createRequest)


if request_list:
    request = {
        "includeFormInResponse": False,
        "requests": request_list
    }

    print(request)

    forms_service.forms().batchUpdate(formId=FORM_ID, body=request).execute()

# sections = form.get('info').get('pageInfo').get('pageElements')
#
# # Find the section with the given name (if it exists)
# section = None
# for s in sections:
#     if s.get('title') == SECTION_NAME:
#         section = s
#         break

# update = {
#     "requests": [{
#         "updateFormInfo": {
#             "info": {
#                 "description": "Please complete this quiz based on this week's readings for class."
#             },
#             "updateMask": "description"
#         }
#     }]
# }
#
# # Define the questions to add to the section
# questions = [
#     {
#         'question': 'What is your name?',
#         'questionType': 'TEXT',
#     },
#     {
#         'question': 'What is your age?',
#         'questionType': 'NUMBER',
#     },
# ]
#
#
# # Replace the section if it exists, otherwise create a new section
# if section_id:
#     requests = [
#         {
#             'updatePageElementProperties': {
#                 'objectId': section_id,
#                 'pageElementProperties': {
#                     'title': SECTION_NAME,
#                     'pageElements': questions,
#                 },
#                 'fields': 'title,pageElements',
#             },
#         },
#     ]
# else:
#     requests = [
#         {
#             'createPageBreak': {
#                 'title': SECTION_NAME,
#             },
#         },
#         {
#             'createPageElement': {
#                 'pageObjectId': FORM_ID,
#                 'elementProperties': {
#                     'title': SECTION_NAME,
#                     'pageElements': questions,
#                 },
#             },
#         },
#     ]
#
# # Send the requests to the API
# try:
#     forms_service.forms().batchUpdate(presentationId=FORM_ID, body={'requests': requests}).execute()
# except HttpError as error:
#     print(f'An error occurred: {error}')