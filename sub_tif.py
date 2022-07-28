import os
# # noinspection PyUnresolvedReferences
# from tuiview import geolinkedviewers

home = "D:\\OWN\\STUDIES\\VIT\\4yr_2sem\\capstone\\Codes\\data\\VIT\\original"
# sub tif
for i in os.listdir(home):
    for ulx in range(0, 13464, 2000):
        for uly in range(0, 13974, 2000):
            os.system('gdal_translate -projwin ' + str(ulx) + ' ' + str(uly) + ' ' + str(ulx + 2000) + ' ' + str(
                uly + 2000) + ' ' + home + '\\' + i + ' D:\\OWN\\STUDIES\\VIT\\4yr_2sem\\capstone\\Codes\\data\\VIT\\demo' + i + '_ulx_' + str(
                ulx) + '_uly_' + str(uly) + '.tif')
