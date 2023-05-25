import datetime
import time
import os
from tqdm import tqdm
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import urllib

from uploadYoutube import YoutubeUpload


class ExtractVideos:
    def __init__(self, driver=None, folder="dat/extract_video", file_names="video_", video_file_extension=".mp4",
                 upload=True, headless=True):
        if not driver:  # Default driver
            op = webdriver.ChromeOptions()
            op.add_argument("--mute-audio")

            if headless:
                op.add_argument('--headless')

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=op)
            # driver.maximize_window()
        self.driver = driver
        self.folder = folder
        self.file_names = file_names
        self.video_file_extension = video_file_extension
        self.current_file_path = None

        # Creating folders if not exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if upload:
            self.yt = YoutubeUpload()

    def _loadPage(self, url, wait_load=0.5):  # Load page if not yet loaded
        if self.driver.current_url != url:
            self.driver.get(url)
            time.sleep(wait_load)  # Wait 0.5 secs

    def extractDataLangs(self, url_eng, url_mlt, upload=True):  # Includes both maltese and english transcripts
        num_files = len([x for x in os.listdir(self.folder) if x.endswith(".json")])  # self.video_file_extension

        name = self.file_names + str(num_files)  # General naming convention
        video_file_path = os.path.join(self.folder, name + self.video_file_extension)  # get path for video file
        text_file_path = os.path.join(self.folder, name + ".json")  # get path for json file

        # Engish
        text_eng = self.extractText(url_eng)
        # Then get video (will load other pages)
        self.extractVideo(url_eng, video_file_path)  # Saves video to file_path

        # Maltese
        text_mlt = self.extractText(url_mlt)

        yt_url = "None"
        if upload:
            line = "----------------"
            divisor = "======================"
            desc = "MALTI\n{}\n".format(line)
            for t in text_mlt.transcript:
                desc += t.__repr__() + "\n\n"
            desc += "{}\n\nENGLISH\n{}\n".format(divisor, line)
            for t in text_eng.transcript:
                desc += t.__repr__() + "\n\n"
            desc += divisor

            if len(desc) > 5000:
                desc = "MALTI\n{}\n".format(line)
                for t in text_mlt.transcript:
                    desc += t.__repr__() + "\n\n"

            if len(desc) > 5000:  # If still longer than 5000
                desc = ""

            try:
                yt_url = self.yt.uploadVideo(video_file_path, name, desc)
            except Exception as e:
                yt_url = str(e)

        # Output json
        output_data = {"MLT": text_mlt.toDict(), "ENG": text_eng.toDict(), "video url": yt_url}
        with open(text_file_path, "w") as f:
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

    def extractText(self, url, file_path=None):  # Extract text from article
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

        # text = self.TextData.listToDict(text)  # Converting to dicts instead of classes
        data_packet = self.PacketData().fillData(title, url, text)
        # data_packet_dict = data_packet.toDict()
        #
        # if file_path:
        #     with open(file_path, "w") as f:
        #         json.dump(data_packet_dict, f, indent=4)

        return data_packet

    def extractVideo(self, url,
                     file_name=None,
                     quality=True,
                     show_updates=False,
                     return_stages=False,
                     links_only=False
                     ):  # Extracts video from link.  Returns the file_path [Warning will load other pages!]

        name = file_name
        if not file_name:  # Sets default file path
            files = [x for x in os.listdir(self.folder) if
                     x.endswith(self.video_file_extension) and x.startswith(self.file_names)]
            num_max = 0
            for file in files:
                num = file.replace(self.video_file_extension, "").replace(self.file_names, "")
                num = int(num)
                if num > num_max:
                    num_max = num

            name = self.file_names + str(num_max + 1)  # Name will be the next number for the default file names
        file_path = os.path.join(self.folder, name + self.video_file_extension)  # get path for video file

        self.current_file_path = file_path

        # Stage 1
        link_1 = self._stage1(url)  # Get first link for video
        if show_updates:
            print("Stage 1: ", link_1)

        # Stage 2
        if quality:
            link_2 = self._stage2Quality(link_1)
        else:
            link_2 = self._stage2(link_1)
        if show_updates:
            print("Stage 2: ", link_2)

        # Stage 3
        if not links_only:
            self._stage3(link_2, file_path)
            if show_updates:
                print("Stage 3: ", file_path)

        if return_stages:
            return link_1, link_2, file_path
        return file_path

    def dateVideoExist(self, date):
        if os.path.exists(os.path.join(self.folder, date + self.video_file_extension)):
            return True
        return False

    def extractNews(self, date, starting_url="https://www.tvmi.mt/mt/tvmi/programmes/l-ahbarijiet-ta-8pm/",
                    quality=True,
                    show_updates=False):
        self._loadPage(starting_url)

        try:
            video_element = self.driver.find_element(By.XPATH, f"//h3[contains(text(), '{date}')]/../../..")
            video_link = video_element.get_attribute("href")
        except NoSuchElementException:
            print("Video does not exist")
            return
        except Exception as e:
            print("Error: ", str(e))
            return

        return self.extractVideo(video_link, file_name=date, quality=quality, show_updates=show_updates)

    def extractAllNews(self, starting_url="https://www.tvmi.mt/mt/tvmi/programmes/l-ahbarijiet-ta-8pm/",
                       quality=True,
                       show_updates=False,
                       duplicates=False,
                       video_links_name="info.json"):

        # Video links
        # { date: {"link 1": link_1, "link 2": link_2, "file name": file_name}}
        LINK0_K = "Normal Link"
        LINK1_K = "Maximised Link"
        LINK2_K = "Video Stream"
        FILENAME_K = "file name"
        video_links_path = os.path.join(self.folder, video_links_name)
        if os.path.exists(video_links_path):
            with open(video_links_path, "r") as f:
                video_links = json.load(f)
        else:
            video_links = dict()

        self._loadPage(starting_url)

        # Searching for episodes
        search_text = "Episodju: "
        video_elements = self.driver.find_elements(By.XPATH, f"//h3[contains(text(), '{search_text}')]")
        if show_updates:
            print("video_elements: ", video_elements)

        # Extracting dates from elements
        date_links = dict()
        for video_element in video_elements:
            # Get link of video
            link_element = video_element.find_element(By.XPATH, "./../../..")
            video_link = link_element.get_attribute("href")

            # Sets date and video link in dictionary
            text = video_element.get_attribute('textContent')
            date = text.replace(search_text, "")
            date_links[date] = video_link
        if show_updates:
            print("date_links:", date_links)

        # Looping over dates and their video links
        file_paths = []
        for date, video_link in date_links.items():
            # If no duplicates allowed and date video already exists, then do not attempt to extract
            if not duplicates and self.dateVideoExist(date):
                if show_updates:
                    print()
                    print(f"Video for date {date} already done")

                # Check if date in video_links, add it if needed
                if date not in video_links:
                    try:
                        link_1, link_2, file_path = self.extractVideo(video_link, file_name=date, quality=quality,
                                                                      show_updates=show_updates, return_stages=True,
                                                                      links_only=True)
                    except Exception:
                        if show_updates:
                            print("Error found, continue")
                        continue

                    file_paths.append(file_path)

                    # Save to json
                    video_links[date] = {LINK0_K: video_link, LINK1_K: link_1, LINK2_K: link_2,
                                         FILENAME_K: date + self.video_file_extension}
                    with open(video_links_path, "w") as f:
                        json.dump(video_links, f, indent=4)

            else:
                if show_updates:
                    print()
                    print(f"Starting extracting for date {date}")

                try:
                    link_1, link_2, file_path = self.extractVideo(video_link, file_name=date, quality=quality,
                                                                  show_updates=show_updates, return_stages=True)
                except Exception:
                    if show_updates:
                        print("Error found, continue")
                    continue
                file_paths.append(file_path)

                # Save to json
                video_links[date] = {LINK0_K: video_link, LINK1_K: link_1, LINK2_K: link_2,
                                     FILENAME_K: date + self.video_file_extension}
                with open(video_links_path, "w") as f:
                    json.dump(video_links, f, indent=4)

        return file_paths

    def extractAllNews2(self, starting_url="https://www.tvmi.mt/mt/tvmi/programmes/l-ahbarijiet-ta-8pm/",
                       ext="?ep=",
                       starting_ep=911950,
                       ending_ep=913731,
                       error_thresh=5,
                       quality=True,
                       show_updates=False,
                       duplicates=False,
                       video_links_name="info.json"):

        pass

    def iterAllNewsUrl(self, starting_url="https://www.tvmi.mt/mt/tvmi/programmes/l-ahbarijiet-ta-8pm/",
                       ext="?ep=",
                       starting_ep=911950,
                       ending_ep=913731,
                       error_thresh=5,
                       quality=True):
        """
        Iterates through all available urls

        :param starting_url:
        :param ext:
        :param starting_ep:
        :param ending_ep:
        :param error_thresh:
        :param quality:
        :return:
        """
        for i in range(starting_ep, ending_ep):
            url = starting_url + ext + str(i)

            try:
                # Stage 1
                link_1 = self._stage1(url)  # Get first link for video
                # Stage 2
                if quality:
                    link_2 = self._stage2Quality(link_1)
                else:
                    link_2 = self._stage2(link_1)

                yield link_2
            except:
                continue

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
            time.sleep(10)  # First wait 9 secs
            skip_elem = self.driver.find_element(By.XPATH, "//div[contains(@class, 'action-countdown-container')]")
            skip_elem.click()
            time.sleep(5)  # Wait again for second ad
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


def extractIteration(json_path, upload=True):
    limit_youtube = 5

    WAITING = 0
    DONE = 1
    DUPLICATE = 2

    EVideos = ExtractVideos()  # Creating class
    with open(json_path) as f:
        to_extract_dat = json.load(f)

    url_set = set()  # For checking duplicates

    count_uploaded_now = 0

    pbar = tqdm(total=len(to_extract_dat), desc="Extracting videos and transcripts of urls")
    for dat in to_extract_dat:
        url_eng = dat["ENG"]
        url_mlt = dat["MLT"]

        if dat["status"] == WAITING:
            # Duplicate handling
            if url_eng in url_set or url_mlt in url_set:  # If any duplicate, skip and mark
                dat["status"] = DUPLICATE
                pbar.update(1)
                continue

            if upload and count_uploaded_now > limit_youtube:
                print("Youtube quota exceeded")
                continue  # Keep skipping

            EVideos.extractDataLangs(url_eng, url_mlt, upload=upload)
            dat["status"] = DONE  # Completed action
            count_uploaded_now += 1

        url_set.add(url_eng)
        url_set.add(url_mlt)

        pbar.update(1)
        # Update file each time
        with open(json_path, "w") as f:  # Finally, output updated status
            json.dump(to_extract_dat, f, indent=4)

# extractIteration("dat/link_extract.json", upload=False)

# EVideos = ExtractVideos(folder="dat/extract_full_news")  # Creating class
# EVideos.extractVideo("https://www.tvmi.mt/mt/tvmi/programmes/l-ahbarijiet-ta-8pm/")
