'''
    basicNMEA.py - This is a basic GPS test using Sixfab's HAT
    Created by Marc Leroy (Pix4D), June 3rd 2019
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

    time.sleep(0.5)
    node.sendATComm("ATE0","OK")
    time.sleep(0.5)
    node.sendATComm("AT+CMEE=2","OK")
    time.sleep(0.5)
    
    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)

    node.sendATComm("AT+QGPSCFG=\"nmeasrc\",1","OK")

    print("Starting loop")
    while True:
        try:
            
            time.sleep(1)
            print(node.getNMEAGGA())
            '''
            time.sleep(1)
            print(node.getNMEARMC())
            '''
            time.sleep(1)
            print(node.getNMEAGSV())
            time.sleep(1)
            print(node.getNMEAGSA())
            '''
            time.sleep(1)
            print(node.getNMEAVTG())
            #ime.sleep(1)
            #rint(node.getFixedLocation())
            '''
            print()

        except KeyboardInterrupt:
            break

    print("\nExited loop, will turn GNSS off and quit.")
    node.turnOffGNSS()
    print("Done. Quiting.\n")


if __name__=='__main__':
    main()
