from zipfile import ZipFile
from shutil import unpack_archive as unpack

# Es muss eine CSV-Datei mit dem Namen "csv_name.zip" im Verzeichnis "dir_name" geben!

# Diese Datei ist gepackte CSV-Datei, die im Original den Namen "csv_name" hatte

# Diese Datei wird dorthin entpackt unter dem Namen "csv_name"

def zip_entpacken (csv_name, dir_name):
    filename = dir_name + "/" + csv_name + ".zip"
    unpack(filename, dir_name)
    data_file = open (dir_name + "/" + csv_name + ".csv", "r")
    return data_file.readlines()