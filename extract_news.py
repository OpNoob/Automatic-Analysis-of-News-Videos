import datetime
import time
import os
from tqdm import tqdm
import json
import validators

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import urllib


class ExtractData:
    def __init__(self, driver=None, headless=True,
                 quality=True, show_updates=False, return_stages=False):
        """

        :param driver: Can set custom driver
        :param headless: Can set headless
        :param quality: Boolean for high quality or default
        :param show_updates: Option to show updates
        :param return_stages: Option to return all stages
        """
        if not driver:  # Default driver
            op = webdriver.ChromeOptions()
            op.add_argument("--mute-audio")

            if headless:
                op.add_argument('--headless')

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=op)
            # driver.maximize_window()
        self.driver = driver

        self.quality = quality
        self.show_updates = show_updates
        self.return_stages = return_stages

    def _loadPage(self, url, wait_load=0.5):
        """
        Load page with timer

        :param url: Url to load
        :param wait_load: Second to wait after load
        :return:
        """
        if self.driver.current_url != url:
            self.driver.get(url)
            time.sleep(wait_load)  # Wait 0.5 secs

    def extractDataLangs(self, transcript_path=None, video_path=None, url_eng=None, url_mlt=None):
        """
        Extracts the transcripts for both english and maltese transcripts, while also downloading the video

        :param video_path: The video path dow the video download
        :param transcript_path: The transcription path for the transcript extraction
        :param url_eng: The english article
        :param url_mlt: The maltese article
        :return:
        """

        assert url_eng is not None or url_mlt is not None

        video_downloaded = False

        # English
        if url_eng is not None:
            text_eng = self.extractText(url_eng)
            text_eng_data = text_eng.toDict()  # Convert to dictionary
            print(url_eng)
            # Then get video (will load other pages)

            if not video_downloaded and video_path is not None:
                try:
                    self.extractVideo(url_eng, video_path)  # Saves video to file_path
                    video_downloaded = True
                except Exception as e:
                    pass
        else:
            text_eng_data = {}

        # Maltese
        if url_mlt is not None:
            text_mlt = self.extractText(url_mlt)
            text_mlt_data = text_mlt.toDict()  # Convert to dictionary

            if not video_downloaded and video_path is not None:
                try:
                    self.extractVideo(url_mlt, video_path)  # Saves video to file_path
                    video_downloaded = True
                except Exception as e:
                    pass
        else:
            text_mlt_data = {}

        # Output json
        if transcript_path is not None:
            output_data = {"Video status": "Downloaded" if video_downloaded else "No Video", "MLT": text_mlt_data,
                           "ENG": text_eng_data}
            with open(transcript_path, "w") as f:
                json.dump(output_data, f, indent=4)

    class TextData:  # Text data packet
        def __init__(self):
            # ENUMS
            self.NORMAL = 0
            self.QUOTE = 1

            self.type = 0
            self.text = None
            self.author = None
            self.title = None

        def typeName(self, type=None):
            if not type:  # If no type specified return the string for the current type
                type = self.type

            if type == self.NORMAL:
                return "NORMAL"
            elif type == self.QUOTE:
                return "QUOTE"
            else:
                return "None"

        def quote(self, text, author):
            self.text = text
            self.author = author
            self.type = self.QUOTE
            return self

        def normal(self, text):
            self.text = text
            self.type = self.NORMAL
            return self

        def __repr__(self):
            if self.type == self.NORMAL:
                return self.text
            if self.type == self.QUOTE:
                return self.text + "\nAuthor: " + self.author
                # return "(Text=\"" + self.text + "\"\tAuthor=\"" + self.author + "\")"

        def toDict(self):  # To maybe use
            return {"text": self.text, "author": self.author, "type": self.type, "type_str": self.typeName()}

        # @staticmethod
        # def listToDict(lst):  # Destructive; Must be in form of [TextData(), TextData(), TextData()]
        #     temp_list = []
        #     for item in lst:
        #         temp_list.append(item.toDict())
        #     return temp_list

    class PacketData:
        def __init__(self):
            self.title = None
            self.url = None
            self.transcript = None
            self.date_extract = datetime.datetime.now()  # Get current date

        def fillData(self, title, url, transcript, date_extract=None):
            if not date_extract:  # Get current date by default
                date_extract = datetime.datetime.now()

            self.title = title
            self.url = url
            self.transcript = transcript
            self.date_extract = date_extract
            return self

        def toDict(self):
            return {"Title": self.title, "url": self.url, "Transcript": [x.toDict() for x in self.transcript],
                    "Extracted On": self.date_extract.strftime("%d/%b/%Y")}

    def extractText(self, url):
        """
        Extracts text from url

        :param url:
        :return:
        """
        self._loadPage(url)

        # Accepts cookies
        try:
            element_accept = self.driver.find_element(By.XPATH, "//a[text()='Accept']")
            element_accept.click()
        except:
            # print("No cookies")
            pass

        # Setting title
        element_title = self.driver.find_element(By.XPATH, "//h1[contains(@class, 'post-title')]")
        title = element_title.text

        text = []
        for elm in self.driver.find_elements(By.XPATH,
                                             "//div[contains(@class, 'inner-post-entry')]/*"):  # Iterating over the inner post (transcript)
            try:
                text_packet = self.TextData()

                if elm.tag_name == "p":  # Normal description
                    if "translations" in elm.get_attribute("class"):  # Skip translation (eng version)
                        continue

                    if elm.text == "":  # Skip empty text
                        continue

                    text.append(text_packet.normal(elm.text))
                elif elm.tag_name == "blockquote":  # Quotes (interviews)
                    text_elem = elm.find_element(By.XPATH, "./p")
                    cite_elem = elm.find_element(By.XPATH, "./cite")
                    text.append(text_packet.quote(text_elem.text, cite_elem.text))
            except:  # skipping stale elements
                pass

        data_packet = self.PacketData().fillData(title, url, text)
        return data_packet

    def extractVideo(self, url, video_path=None):
        """
        Extracts video from url
        Warning: Will load other pages

        :param url: Video url
        :param video_path: Video path to extract
        :return:
        """

        # Stage 1
        link_1 = self._stage1(url)  # Get first link for video
        if self.show_updates:
            print("Stage 1: ", link_1)

        # Stage 2
        if self.quality:
            link_2 = self._stage2Quality(link_1)
        else:
            link_2 = self._stage2(link_1)
        if self.show_updates:
            print("Stage 2: ", link_2)

        if video_path is not None:
            self._stage3(link_2, video_path)
            if self.show_updates:
                print("Stage 3: ", video_path)

        if self.return_stages:
            return link_1, link_2, video_path

    def _stage1(self, url):  # Get html from media.tvm player
        self._loadPage(url)

        search_for = "media.tvm"
        elements = self.driver.find_elements(By.XPATH, "//iframe")

        if len(elements) == 0:
            print("No video for url:", url)

        for element in elements:
            link = element.get_attribute("src")
            if link:
                if search_for in link:
                    return link

    def _stage2(self, url):  # Get link from media.tvm files
        self._loadPage(url)

        # Starts video
        click_elem = self.driver.find_element(By.XPATH, "//img")
        click_elem.click()

        # Then searches for next link
        search_for = "media.tvm"
        element = self.driver.find_element(By.XPATH, "//video")
        # element.click()
        link = element.get_attribute("src")
        if not link:
            print("No link found")
            return
        if search_for not in link:
            print("Link not in correct format")
            return
        return link

    def _stage2Quality(self, url):  # Get link from media.tvm files
        self._loadPage(url)

        # Starts video
        click_elem = self.driver.find_element(By.XPATH, "//img")
        click_elem.click()

        # Skip ad [Might fail here]
        try:
            time.sleep(10)  # First wait
            skip_elem = self.driver.find_element(By.XPATH, "//div[contains(@class, 'action-countdown-container')]")
            skip_elem.click()
            time.sleep(10)  # Wait again for second ad
            skip_elem = self.driver.find_element(By.XPATH, "//div[contains(@class, 'action-countdown-container')]")
            skip_elem.click()
        except Exception as e:
            # print("No skip: ", str(e))
            print("No skipping")

        # Select quality
        # Open quality options
        quality_elem = self.driver.find_element(By.XPATH, "//button[contains(@class, 'quality')]")
        quality_elem.click()
        time.sleep(0.5)
        quality_elem.send_keys(Keys.TAB)

        # Choose hd
        hd_elem = self.driver.find_element(By.XPATH, "//button[contains(@label, 'hd')]")
        hd_elem.click()

        # Then searches for next link
        search_for = "media.tvm"
        element = self.driver.find_element(By.XPATH, "//video")
        # element.click()
        link = element.get_attribute("src")
        if not link:
            print("No link found")
            return
        if search_for not in link:
            print("Link not in correct format")
            return
        return link

    def _stage3(self, url, file_path=None, show_progress=True):  # Download mp4 video
        if not file_path:
            num_files = len([x for x in os.listdir(self.folder) if x.endswith(self.video_file_extension)])
            name = self.file_names + str(num_files)  # General naming convention
            file_path = os.path.join(self.folder, name + self.video_file_extension)  # get path for video file

        if show_progress:  # https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
            class MyProgressBar:
                def __init__(self, link, path):
                    self.pbar = None
                    self.link = link
                    self.path = path

                def __call__(self, block_num, block_size, total_size):
                    if not self.pbar:
                        self.pbar = tqdm(total=total_size, desc="Downloading...")
                        # self.pbar.start()

                    downloaded = block_num * block_size
                    if downloaded < total_size:
                        self.pbar.update(block_size)
                    else:
                        self.pbar.close()

            urllib.request.urlretrieve(url, file_path, MyProgressBar(url, file_path))
        else:
            urllib.request.urlretrieve(url,
                                       file_path)  # https://stackoverflow.com/questions/48736437/how-to-download-this-video-using-selenium

    def __del__(self):  # Closes driver when class is deleted
        self.driver.close()


def ArticleLinksExtract(input_folder="data/Article_Links", output_folder="data/Article_Links_Extraction",
                        extract_data=None):
    os.makedirs(output_folder, exist_ok=True)

    if extract_data is None:
        extract_data = ExtractData()

    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for month in folders:
        json_files = [f for f in os.listdir(os.path.join(input_folder, month)) if f.endswith(".json")]

        pbar = tqdm(total=len(json_files), desc=f"Extracting from folder '{os.path.join(input_folder, month)}'")
        for json_file_name in json_files:
            json_path = os.path.join(input_folder, month, json_file_name)
            with open(json_path, "r") as f:
                data_dict = json.load(f)
                transcripts = data_dict["Transcripts"]
                for t in transcripts:
                    url_mlt = t["URL MLT"]
                    url_eng = t["URL ENG"]
                    if validators.url(url_mlt):
                        try:
                            transcript_mlt = extract_data.extractText(url_mlt).toDict()
                        except Exception as e:
                            e_str = str(e)
                            print(e_str)
                            transcript_mlt = {"ERRROR": e_str}
                    else:
                        transcript_mlt = {}
                    if validators.url(url_eng):
                        try:
                            transcript_eng = extract_data.extractText(url_eng).toDict()
                        except Exception as e:
                            e_str = str(e)
                            print(e_str)
                            transcript_eng = {"ERRROR": e_str}
                    else:
                        transcript_eng = {}
                    t["transcript mlt"] = transcript_mlt
                    t["transcript eng"] = transcript_eng
            json_path_new = os.path.join(output_folder, json_file_name)
            with open(json_path_new, "w") as fo:
                json.dump(data_dict, fo, indent=4)
            pbar.update()
        pbar.close()


if __name__ == "__main__":
    ed = ExtractData(headless=False, quality=True, show_updates=True)
    ed.extractDataLangs(transcript_path="tests/test.json", video_path="tests/test.mp4",
                        url_mlt="https://tvmnews.mt/news/il-popolazzjoni-tizdied-b25-fl-ahhar-10-snin-ghal-519562/",
                        url_eng="https://tvmnews.mt/en/news/population-increases-by-25-over-last-10-years-to-519562/")

    # ed = ExtractData(headless=False, quality=True, show_updates=True)
    # ArticleLinksExtract(extract_data=ed)
