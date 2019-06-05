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

    '''
    print("Getting HW info")
    node.getHardwareInfo()
    print("Done")
    time.sleep(1)
    #rint("Sending ATE1 command")
    #ode.sendATComm("ATE1", "OK")
    '''

    print("Enabling gpsOneXTRA Assistance")
    node.sendATComm("AT+QGPSXTRA=1", "OK")
    node.sendATComm("AT+QGPSXTRADATA?","QGPSXTRADATA")
    print("Enabled\n")
    time.sleep(0.5)
    
    geofence_lat = 46.529303
    geofence_lon = 6.601181
    geofence_rad0 = 100
    geofence_rad1 = 500
    geofence_rad2 = 1000
    geofence_rad3 = 5000

    print("Setting a circular geo-fence centered around "+str(geofence_lat)+"N, "+str(geofence_lon)+"E of radius "+str(geofence_rad0)+"[m]")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",0,3,0,"+str(geofence_lat)+","+str(geofence_lon)+","+str(geofence_rad0), "OK")
    time.sleep(0.5)
    print("Checking it")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",0", "QCFGEXT")
    print("Geo-fence 0 set")

    print("Setting a circular geo-fence centered around "+str(geofence_lat)+"N, "+str(geofence_lon)+"E of radius "+str(geofence_rad1)+"[m]")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",1,3,0,"+str(geofence_lat)+","+str(geofence_lon)+","+str(geofence_rad1), "OK")
    time.sleep(0.5)
    print("Checking it")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",1", "QCFGEXT")
    print("Geo-fence 1 set")
    
    print("Setting a circular geo-fence centered around "+str(geofence_lat)+"N, "+str(geofence_lon)+"E of radius "+str(geofence_rad2)+"[m]")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",2,3,0,"+str(geofence_lat)+","+str(geofence_lon)+","+str(geofence_rad2), "OK")
    time.sleep(0.5)
    print("Checking it")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",2", "QCFGEXT")
    print("Geo-fence 2 set")

    print("Setting a circular geo-fence centered around "+str(geofence_lat)+"N, "+str(geofence_lon)+"E of radius "+str(geofence_rad3)+"[m]")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",3,3,0,"+str(geofence_lat)+","+str(geofence_lon)+","+str(geofence_rad3), "OK")
    time.sleep(0.5)
    print("Checking it")
    node.sendATComm("AT+QCFGEXT=\"addgeo\",3", "QCFGEXT")
    print("Geo-fence 3 set")

    print("Turning GNSS on")
    node.turnOnGNSS()
    time.sleep(1)
    
    print("Starting loop")
    ctr = 0
    while ctr<1000:
        ctr+=1
        try:
            node.getFixedLocation()
            time.sleep(0.5)
            node.sendATComm("AT+QCFGEXT=\"querygeo\",0", "QCFGEXT")
            node.sendATComm("AT+QCFGEXT=\"querygeo\",1", "QCFGEXT")
            node.sendATComm("AT+QCFGEXT=\"querygeo\",2", "QCFGEXT")
            node.sendATComm("AT+QCFGEXT=\"querygeo\",3", "QCFGEXT")
            time.sleep(0.5)

        except KeyboardInterrupt:
            break

    print("\nExited loop, will turn GNSS off and quit.")
    node.turnOffGNSS()
    time.sleep(0.5)
    node.sendATComm("AT+QCFGEXT=\"deletegeo\",0", "OK")
    node.sendATComm("AT+QCFGEXT=\"deletegeo\",1", "OK")
    node.sendATComm("AT+QCFGEXT=\"deletegeo\",2", "OK")
    node.sendATComm("AT+QCFGEXT=\"deletegeo\",3", "OK")
    print("Done. Quiting.\n")


if __name__=='__main__':
    main()
