import server_video_binary
#import server_ultra
#import server_microphone
import server_steer
import server_socket
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import threading

if __name__ == '__main__':
    # host, port
    host, port = "141.223.122.69", 8000
    print("1")
    client = server_socket.Server(host, port)
    print("2")

    steer = server_steer.Steer(client.Get_Client())
    
#-----------------------------------------------------------------------------------
    print("3")

#    ultrasonic_object = server_ultra.UltraSonic(host, port+1, steer)
#    ultrasonic_object.Run()


#    microphone_object = server_microphone.Microphone(host, port+2, steer)
#    microphone_object.Run()

    # loop

    video_object = server_video_binary.CollectTrainingBinaryData(client.Get_Client(), steer)
    video_object.collect()



#rc_driver_main -> server_socket -> steer_server -> server_video (while loop)
#rc_driver_main -> server_socket -> steer_server -> server_microphone (while loop)
#rc_driver_main -> server_socket -> steer_server -> server_ultra (while loop)
