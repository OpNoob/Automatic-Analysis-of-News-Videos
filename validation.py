import json
import os
from tqdm import tqdm
from utils import parseTimestamp

"""
** DEPRECATED **

Validation

Which people to expect in different segments.
Includes both maltese and english transcripts since some names are written differently
Also only considers the names, although face recognition is taken into account when matching with preexisting people 
"""

START_S_K = "start_s"
END_S_K = "end_s"
PEOPLE_ENG_K = "people_eng"
PEOPLE_MLT_K = "people_mlt"
URL_MLT_K = "URL MLT"
URL_ENG_K = "URL ENG"


def getTranscriptData(transcript: dict, name=None, time_stamp_parser=parseTimestamp):
    # Check if no transcripts
    if not transcript["transcript mlt"] and not transcript["transcript eng"]:
        return

    # Get timestamp in seconds
    time_stamp = transcript["Time stamp"]
    time_stamp_s = time_stamp_parser(time_stamp, name=name)
    if time_stamp_s is False:
        return

    # Get end timestamp
    duration = transcript["Duration"]
    duration_s = time_stamp_parser(duration, name=name)
    if duration_s is False:
        return

    url_mlt = transcript["URL MLT"]
    url_eng = transcript["URL ENG"]

    time_stamp_end_s = time_stamp_s + duration_s  # Calculate end timestap

    # Add this data
    people_list_eng = list()
    people_list_mlt = list()
    data_dict = {START_S_K: time_stamp_s,
                 END_S_K: time_stamp_end_s,
                 PEOPLE_ENG_K: people_list_eng,
                 PEOPLE_MLT_K: people_list_mlt,
                 URL_MLT_K: url_mlt,
                 URL_ENG_K: url_eng,
                 }

    # Fill people list
    for transcript_type, plist in [("transcript mlt", people_list_mlt),
                                   ("transcript eng", people_list_eng)]:
        transcripts_dict = transcript[transcript_type]
        if "Transcript" in transcripts_dict:
            for segment in transcripts_dict["Transcript"]:
                author = segment["author"]
                if author is not None:
                    plist.append(author)

    return data_dict


def getValidation(dir_path, output_path=None):
    ret_dict = dict()  # return dictionary

    files = [f for f in os.listdir(dir_path) if f.endswith(".json")]  # get json files
    pbar = tqdm(total=len(files), desc="Retrieving meaningful data for validation")
    for json_file_name in files:
        name = os.path.splitext(json_file_name)[0]
        json_path = os.path.join(dir_path, json_file_name)
        with open(json_path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)

            # Verification that names match
            news_date = json_dict["News date"]
            if news_date != name:
                print(f"Name '{name}' and News date '{news_date}' did not Match!")

            # Creating dictionary for current iteration
            event_list = list()
            ret_dict[name] = event_list

            transcripts = json_dict["Transcripts"]
            for transcript in transcripts:
                data_dict = getTranscriptData(transcript, name=name)
                if data_dict:
                    event_list.append(data_dict)

                # # Check if no transcripts
                # if not transcript["transcript mlt"] and not transcript["transcript eng"]:
                #     continue
                #
                # # Get timestamp in seconds
                # time_stamp = transcript["Time stamp"]
                # time_stamp_s = parseTimestamp(time_stamp, name=name)
                # if time_stamp_s is False:
                #     continue
                #
                # # Get end timestamp
                # duration = transcript["Duration"]
                # duration_s = parseTimestamp(duration, name=name)
                # if duration_s is False:
                #     continue
                #
                # url_mlt = transcript["URL MLT"]
                # url_eng = transcript["URL ENG"]
                #
                # time_stamp_end_s = time_stamp_s + duration_s  # Calculate end timestap
                #
                # # Add this data
                # people_list_eng = list()
                # people_list_mlt = list()
                # event_list.append(
                #     {START_S_K: time_stamp_s,
                #      END_S_K: time_stamp_end_s,
                #      PEOPLE_ENG_K: people_list_eng,
                #      PEOPLE_MLT_K: people_list_mlt,
                #      URL_MLT_K: url_mlt,
                #      URL_ENG_K: url_eng,
                #      }
                # )
                #
                # # Fill people list
                # for transcript_type, plist in [("transcript mlt", people_list_mlt),
                #                                ("transcript eng", people_list_eng)]:
                #     transcripts_dict = transcript[transcript_type]
                #     if "Transcript" in transcripts_dict:
                #         for segment in transcripts_dict["Transcript"]:
                #             author = segment["author"]
                #             if author is not None:
                #                 plist.append(author)

        pbar.update()
    pbar.close()

    if output_path is not None:
        dir_create = os.path.dirname(output_path)
        os.makedirs(dir_create, exist_ok=True)
        with open(output_path, "w") as fo:
            json.dump(ret_dict, fo, indent=4)

    return ret_dict


if __name__ == "__main__":
    valid = getValidation("data/Article_Links_Extraction", output_path="data/Validation/validation.json")
    print(valid)
