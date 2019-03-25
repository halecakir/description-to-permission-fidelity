import os
import sys
import inspect
import csv


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)



def apps_with_given_permission(file_path, included_permission):
    """TODO"""
    number_of_apps = 0
    all_permissions = set()
    apps_with_given_permission = 0
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
                all_permissions.add(permission)
    return number_of_apps, len(all_permissions), apps_with_given_permission, all_permissions

if __name__ == "__main__":
    DIR_NAME = os.path.dirname(__file__)
    IN_PATH = os.path.join(DIR_NAME, "../../../data/big_processed/_apps_processed.csv")
    APP_COUNT, PERMISSION_COUNT, COUNT_APP_WITH_GIVEN_PERM, PERMISSIONS = apps_with_given_permission(IN_PATH, "READ_CALENDAR")
    print("Total Applications {}\nTotal Distinct Permissions {}\nApplication with given permisssion {}\n".format(APP_COUNT, PERMISSION_COUNT, COUNT_APP_WITH_GIVEN_PERM))
    for perm in PERMISSIONS:
        print(perm)