import copy
import os
from tqdm import tqdm
import plots

import validation

import validation
from video_analysis_1 import *

"""
Old comparison with old validation
"""

valid = validation.getValidation("data/Article_Links_Extraction")


class Validation:
    class EventAnalysis:
        def __init__(self, people_checkoff=None, people_checkon=None, people_extra=None, data=None):
            if people_extra is None:
                people_extra = set()
            if people_checkon is None:
                people_checkon = set()
            if people_checkoff is None:
                people_checkoff = set()
            if data is None:
                data = dict()
            self.people_checkoff = people_checkoff
            self.people_checkon = people_checkon
            self.people_extra = people_extra
            self.data = data

        def toDict(self):
            return {
                "people_checkoff": list(self.people_checkoff),
                "people_checkon": list(self.people_checkon),
                "people_extra": list(self.people_extra),
                "data": self.data,
            }

    def __init__(self, folders_located=None):
        if folders_located is None:
            folders_located = list()
        self.folders_completed = list()
        self.folders_uncompleted = copy.deepcopy(folders_located)
        self.data = list()

    def add(self, data: EventAnalysis):
        self.data.append(data)

    def toDict(self):
        return [d.toDict() for d in self.data]

    def getCM(self, graph_path=None, name: str=None):
        people_checkoff = list()  # FN
        people_checkon = list()  # TP
        people_extra = list()  # FP
        nothing = list()  # TN

        for d in self.data:
            people_checkoff += d.people_checkoff
            people_checkon += d.people_checkon
            people_extra += d.people_extra

        cm = plots.ConfusionMatrix("Confusion Matrix" if name is None else name, save_location=graph_path)
        cm.fn = len(people_checkoff)
        cm.tp = len(people_checkon)
        cm.fp = len(people_extra)
        cm.tn = len(nothing)

        cm.genPlot()


def getComparison(directory, output_data_dir="data/comparing/results",
                  analysis_data_name="analysis_data"):
    os.makedirs(output_data_dir, exist_ok=True)
    folder_name = os.path.basename(directory)
    log_path = os.path.join(output_data_dir, folder_name + ".json")
    graph_path = os.path.join(output_data_dir, folder_name + ".png")

    folders_located = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    validation_data = Validation(folders_located=folders_located)
    # folders_completed = list()
    # folders_uncompleted = copy.deepcopy(folders_located)

    pbar = tqdm(total=len(valid), desc="Validating")
    for name, events in valid.items():
        base_dir = os.path.join(directory, name)

        if os.path.exists(base_dir):
            # Set lists
            validation_data.folders_completed.append(name)
            validation_data.folders_uncompleted.remove(name)

            na = NewsAnalysis(None, base_dir, analysis_data_name=analysis_data_name,
                              load_previous=True)  # Load analysis data
            scene_iter = iter(na.scenes)
            current_scene = next(scene_iter, None)
            if current_scene is None:
                continue  # Continue to other video analysis
            start_scene_s = current_scene.scene_timestamps[0].get_seconds()
            end_scene_s = current_scene.scene_timestamps[1].get_seconds()

            for current_event in events:
                start_s = current_event[validation.START_S_K]
                end_s = current_event[validation.END_S_K]
                people_eng = current_event[validation.PEOPLE_ENG_K]
                people_mlt = current_event[validation.PEOPLE_MLT_K]
                url_mlt = current_event[validation.URL_MLT_K]
                url_eng = current_event[validation.URL_ENG_K]

                people_checkoff = set(
                    copy.deepcopy(people_eng) + copy.deepcopy(people_mlt))  # Removed people that are seen (FP)
                people_checkoff_copy = copy.deepcopy(people_checkoff)  # Holds a copy of people checkoff
                people_checkon = set()  # Adds people that are seen (TP)

                people_extra = set()  # List of people that are in scenes that should not be (FP)
                event_data = Validation.EventAnalysis(people_checkoff=people_checkoff, people_extra=people_extra,
                                                      people_checkon=people_checkon, data=current_event)
                validation_data.add(event_data)

                # Consume until match scene starts inside the event
                while not start_s <= start_scene_s:
                    current_scene = next(scene_iter, None)
                    if current_scene is None:
                        break
                    start_scene_s = current_scene.scene_timestamps[0].get_seconds()
                    end_scene_s = current_scene.scene_timestamps[1].get_seconds()
                if current_scene is None:
                    continue  # Continue to other video analysis

                while start_s <= start_scene_s and end_scene_s <= end_s:
                    face_info = current_scene.face_info
                    if face_info is not None:
                        name = face_info.name
                        if name is not None:
                            name = name.lower()
                            # print(name, start_scene_s, end_scene_s, start_s, end_s)

                            at_least_one_match = False
                            for p in people_checkoff_copy:
                                if name in p.lower():
                                    people_checkon.add(name)  # Adds name with found one
                                    if p in people_checkoff:
                                        people_checkoff.remove(p)  # Removed the name
                                    at_least_one_match = True  # Marks at least one match
                            if not at_least_one_match:
                                people_extra.add(name)

                    current_scene = next(scene_iter, None)
                    if current_scene is None:
                        break
                    start_scene_s = current_scene.scene_timestamps[0].get_seconds()
                    end_scene_s = current_scene.scene_timestamps[1].get_seconds()
                if current_scene is None:
                    continue  # Continue to other video analysis

                print("\n")
                # print(start_s, end_s)
                print("people_checkoff", people_checkoff)
                print("people_checkon", people_checkon)
                print("people_extra", people_extra)
                print("\n")

        pbar.update()
    pbar.close()

    with open(log_path, "w") as f:
        json.dump(validation_data.toDict(), f, indent=4)

    validation_data.getCM(graph_path=graph_path, name=folder_name + " Confusion Matrix")

    return validation_data


if __name__ == "__main__":
    # log1 = getComparison("data/comparing/encoding no database 1")
    log2 = getComparison("data/comparing/encoding with database 1")
    # print(log.toDict())
