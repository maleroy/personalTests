'''
    logWithGpsOneXTRA.py - This is a GNSS test using Sixfab's HAT with gpsOneXTRA Assistance Function from Qualcomm that logs the GPS info in a file
    Created by Marc Leroy (Pix4D), May 8th 2019
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

    print("Getting HW info")
    node.getHardwareInfo()
    print("Done")
    time.sleep(1)
    #rint("Sending ATE1 command")
    #ode.sendATComm("ATE1", "OK")

    print("Enabling gpsOneXTRA Assistance")
    node.sendATComm("AT+QGPSXTRA=1", "OK")
    node.sendATCommOnce("AT+QGPSXTRADATA?")
    print("Enabled\n")
    time.sleep(0.5)
    
    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)
    
    print("Starting loop")
    ctr = 0
    while ctr<1000:
        ctr+=1
        try:
            #ode.sendATCommOnce("AT+QGPSXTRADATA?")
            node.getFixedLocation()
            time.sleep(1)

        except KeyboardInterrupt:
            break

    print("\nExited loop, will turn GNSS off and quit.")
    node.turnOffGNSS()
    print("Done. Quiting.\n")


if __name__=='__main__':
    main()
