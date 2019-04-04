import requests
import time


start_url = 'http://localhost:3000/api/apps/?collection=topselling_free&category=COMMUNICATION&lang=en'
waiting_urls = set()
app_ids = set()

#top free communication
response = requests.get(start_url)
if response.status_code == 200:
    for app in response.json()["results"]:
        waiting_urls.add(app["url"])
else:
    exit("Request Error")

counter = 0
start_time = time.time()
while (len(app_ids) + len(waiting_urls)) < 100000:
    if counter % 60 == 0:
        elapsed_time = time.time() - start_time
        print("Total number of apps", len(app_ids) + len(waiting_urls))
        print("Collected app ids", len(app_ids))
        print("Elapsed time up to now is {}".format(elapsed_time))
    try:
        url = waiting_urls.pop()
    except KeyError:
        print("All linked applications are traversed")
        break

    #add apk id if it is free, popular, and has longer description than 100 characters
    response = requests.get(url)
    if response.status_code == 200:
        json = response.json()
        if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
            app_ids.add(url.split('/')[-1])

    #add similar app urls
    response = requests.get(url + '/similar')
    if response.status_code == 200:
        for app in response.json()["results"]:
            if app["appId"] not in app_ids:
                waiting_urls.add(app["url"])
    counter += 1

for url in waiting_urls:
    response = requests.get(url)
    if response.status_code == 200:
        json = response.json()
        if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
            app_ids.add(url.split('/')[-1])
    app_ids.add(url.split('/')[-1])

with open("app_ids.txt", "w") as target:
    for app in app_ids:
        target.write(app+"\n")
