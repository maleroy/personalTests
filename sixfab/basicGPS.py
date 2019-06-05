'''
    basicGPS.py - This is a basic GPS test using Sixfab's HAT
    Created by Marc Leroy (Pix4D), May 3rd 2019
'''
from cellulariot import cellulariot
import time

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
