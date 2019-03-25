import os
import sys
import inspect
import csv
import operator

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)



def apps_with_given_permission(file_path, included_permission):
    """TODO"""
    number_of_apps = 0
    apps_with_given_permission = 0
    permission_statistics = {}
    with open(file_path) as stream:
        reader = csv.reader(stream)
        header = next(reader)
        for row in reader:
            title = row[0]
            text = row[1]
            permissions = row[2]
            link = row[3]

            app_perms = {perm for perm in permissions.split("%%")}
            number_of_apps += 1
            if included_permission in app_perms:
                apps_with_given_permission += 1
            for permission in app_perms:
                if permission not in permission_statistics:
                    permission_statistics[permission] = 0
                permission_statistics[permission] += 1
    return number_of_apps, apps_with_given_permission, permission_statistics

if __name__ == "__main__":
    DIR_NAME = os.path.dirname(__file__)
    IN_PATH = os.path.join(DIR_NAME, "../../../data/big_processed/apps_processed.csv")
    APP_COUNT, COUNT_APP_WITH_GIVEN_PERM, PERMISSION_STATISTICS = apps_with_given_permission(IN_PATH, "READ_CALENDAR")
    print("Total Applications {}\nTotal Distinct Permissions {}\n".format(APP_COUNT, len(PERMISSION_STATISTICS)))
    sorted_permission_stats = sorted(PERMISSION_STATISTICS.items(), key=operator.itemgetter(1), reverse=True)
    for rank, pair in enumerate(sorted_permission_stats):
        print("{}.{} :: {}\n".format(rank+1, pair[0], pair[1]))
