import requests
import zipfile
import io

def get_qualtrics_survey(dir_save_survey, survey_id):
    """ automatically query the qualtrics survey data
    guide https://community.alteryx.com/t5/Alteryx-Designer-Discussions/Python-Tool-Downloading-Qualtrics-Survey-Data-using-Python-API/td-p/304898 """

    # Setting user Parameters
    api_token = "S4D7SsvQvFco9hWlyAUyZonJeS5NNotMV3emkTMq"
    file_format = "csv"
    data_center = 'usc.ca1' # "<Organization ID>.<Datacenter ID>"

    # Setting static parameters
    request_check_progress = 0
    progress_status = "in progress"
    base_url = "https://{0}.qualtrics.com/API/v3/responseexports/".format(data_center)
    headers = {
        "content-type": "application/json",
        "x-api-token": api_token,
    }

    # Step 1: Creating Data Export
    download_request_url = base_url
    download_request_payload = '{"format":"' + file_format + '","surveyId":"' + survey_id + '"}' # you can set useLabels:True to get responses in text format
    download_request_response = requests.request("POST", download_request_url, data=download_request_payload, headers=headers)
    progress_id = download_request_response.json()["result"]["id"]
    # print(download_request_response.text)

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while request_check_progress < 100 and progress_status != "complete":
        request_check_url = base_url + progress_id
        request_check_response = requests.request("GET", request_check_url, headers=headers)
        request_check_progress = request_check_response.json()["result"]["percentComplete"]

    # Step 3: Downloading file
    request_download_url = base_url + progress_id + '/file'
    request_download = requests.request("GET", request_download_url, headers=headers, stream=True)

    # Step 4: Unzipping the file
    zipfile.ZipFile(io.BytesIO(request_download.content)).extractall(dir_save_survey)
    print('Downloaded qualtrics survey')

if __name__ == "__main__":

    path = ""
    learning_survey_id = "SV_8eoX63z06ZhVZRA"

    get_qualtrics_survey(dir_save_survey = path, survey_id = learning_survey_id)