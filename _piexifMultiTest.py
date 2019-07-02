import os
import sys
import piexif

KEY_MEANINGS = {}
KEY_MEANINGS['0th'] = {34853: 'GPS Info IFD Pointer', 296:'Resolution Unit', 34665:'Exif IFD Pointer',
                       272: 'Model', 305: 'Software', 274: 'Orientation', 271: 'Make', 282: 'XResolution',
                       283: 'YResolution', 315: 'Artist', 306: 'DateTime', 270: 'Image Description', 
                       33432: 'Copyright',
                      }
KEY_MEANINGS['1st'] = {514:'JPEGInterchangeFormatLength', 513:'JPEGInterchangeFormat',
                       296:'ResolutionUnit', 282:'XResolution', 283:'YResolution',
                       37393: 'ImageNumber'
                      }
KEY_MEANINGS['GPS'] = {0:'GPSVersionID', 1:'GPSLatitudeRef', 2:'GPSLatitude', 3:'GPSLongitudeRef',
                       4:'GPSLongitude', 5:'GPSAltitudeRef', 6:'GPSAltitude', 7:'GPSTimeStamp',
                       9:'GPSStatus', 18:'GPSMapDatum', 29:'GPSDateStamp',
                      }

KEY_MEANINGS['Exif'] = {40963:'PixelYDimension', 34850:'ExposureProgram', 36867:'DateTimeOriginal',
                        37380:'ExposureBiasValue', 37381:'MaxApertureValue', 37382:'SubjectDistance',
                        34855:'PhotographicSensitivity', 37386:'FocalLength', 37378:'ApertureValue',
                        41486:'FocalPlaneXResolution', 41487:'FocalPlaneYResolution',
                        41488:'FocalPlaneResolutionUnit', 37521:'SubSecTimeOriginal', 40962:'PixelXDimension',
                        41987:'WhiteBalance', 42033:'BodySerialNumber', 42036:'LensModel',
                        33434:'ExposureTime', 33437:'FNumber', 41989:'FocalLengthIn35mmFilm',
                      }

def getFiles(path):
    fileType = path.split(".")[-1]
    directory = path.split("*")[0]
    return [i for i in os.listdir(directory) if fileType in i.split(".")[-1]]

def extract_exif(myFile, saveTbnl):
    # Get base image
    print("Handling "+str(myFile)+" now.")
    myImgExif = piexif.load(myFile)

    # Get thumbnail from base image
    tbnl = myImgExif.pop("thumbnail")
    if tbnl is not None and saveTbnl:
        with open("tbnl_"+myFile, "wb+") as f:
            f.write(tbnl)

    # Remove Exif tags (either non standard or gibberish)
    toRemove = {'0th': [50735, 50708], 'Exif': [37500]}
    for k,v in toRemove.items():
        for i in range(len(v)):
            try:
                myImgExif[k].pop(v[i])
            except KeyError:
                pass  # print("Tag {}-{} already removed".format(k,v[i]))

    # Print 
    for ifd_name in sorted(myImgExif):
        if ifd_name=='0th':
            print("\n{} IFD:".format(ifd_name))
        for key in sorted(myImgExif[ifd_name]):
            if key == 270:
                print(str(key).rjust(5), KEY_MEANINGS[ifd_name][key].rjust(27), '---', myImgExif[ifd_name][key])

    print()
    saveImg = None
    while(not (saveImg=='y' or saveImg=='n' or saveImg=='q')):
        saveImg = input('Save new ' + str(myFile) + ' with updated exif? (y/n) (q to quit): ')
    
    if saveImg=='y':
        myImgExifBytes = piexif.dump(myImgExif)
        os.system('cp ' + myFile + ' out.jpg')
        outExif = piexif.load('out.jpg')
        print('\n\nOutExif_Og\n')
        for ifd_name in outExif:
            print("\n{} IFD:".format(ifd_name))
            for key in outExif[ifd_name]:
                print(key, outExif[ifd_name][key])

        piexif.insert(myImgExifBytes, "out.jpg")        
        
        outExif = piexif.load('out.jpg')
        outExif.pop("thumbnail")
        print('\n\nOutExif_Upd\n')
        for ifd_name in outExif:
            print("\n{} IFD:".format(ifd_name))
            for key in outExif[ifd_name]:
                print(key, outExif[ifd_name][key])

    elif saveImg=='q':
        print('Understood. Quitting program now.')
        sys.exit()

    else:
        print('Done reading this image.')
    
    print('\n\n\n')

def main():
    if len(sys.argv)<2:
        print("Error, provide at least one image")
        sys.exit()

    saveTbnl = False

    myFiles = getFiles(sys.argv[1])
    print(myFiles)

    for i in myFiles:
        extract_exif(i, saveTbnl)

    print('Done reading all images. Bye.')

if __name__=='__main__':
    main()
