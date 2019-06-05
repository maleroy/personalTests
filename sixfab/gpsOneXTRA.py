'''
    gpsOneXTRA.py - This is a GNSS test using Sixfab's HAT with gpsOneXTRA Assistance Function from Qualcomm
    Created by Marc Leroy (Pix4D), May 7th 2019
'''
from cellulariot import cellulariot
import time


def main():
    node = cellulariot.CellularIoT()
    node.setupGPIO()

    print("Power up sequence - disabling first")
    node.disable()
    print("Disable done\n")
    time.sleep(1)
    print("Starting enable")
    node.enable()
    print("Enable done\n")
    time.sleep(1)
    print("Starting power up")
    node.powerUp()
    print("Power up done\n")

    print("Sending ATE1 command")
    node.sendATComm("ATE1", "OK")

    print("Getting IMEI")
    node.getIMEI()
    print()
    time.sleep(0.5)

    print("Getting Firmware")
    node.getFirmwareInfo()
    time.sleep(0.5)

    print("Getting HW Info")
    node.getHardwareInfo()
    time.sleep(0.5)

    print("Enabling gpsOneXTRA Assistance")
    node.sendATComm("AT+QGPSXTRA=1", "OK")
    print("Enabled\n")
    time.sleep(0.5)
    '''
    print("Uploading gpsOneXTRA file via QCOM")
    node.sendATCommOnce("AT+QFUPL=\"UFS:xtra2.bin\",60831,60")
    print("Uploaded?\n")
    time.sleep(60)
    node.sendATCommOnce("AT+QFUPL=?")
    time.sleep(5)

    print("Injecting gpsOneXTRA time to GNSS engine")
    one_xtra_time = time.strftime("%Y/%m/%d,%H:%M:%S",time.gmtime())
    node.sendATComm("AT+QGPSXTRATIME=0,\"" + one_xtra_time + "\",1,1,5", "OK")
    print("Done\n")
    time.sleep(0.5)

    print("Injecting gpsOneXTRA data to GNSS engine")
    node.sendATComm("AT+QGPSXTRADATA=\"UFS:xtra2.bin\"", "OK")
    print("Done\n")
    time.sleep(0.5)

    print("Deleting gpsOneXTRA file from UFS file")
    node.sendATComm("AT+QFDEL=\"UFS:xtra2.bin\"", "OK")
    print("Done\n")
    time.sleep(0.5)
    '''
    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)

    print("Starting loop")
    while True:
        try:
            print(node.getLongitude())
            time.sleep(1)

        except KeyboardInterrupt:
            break

    print("\nExited loop, will turn GNSS off and quit.")
    node.turnOffGNSS()
    print("Done. Quiting.\n")


if __name__=='__main__':
    main()
