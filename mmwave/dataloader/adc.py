#############################################
# Author: Luca Yu
# Date: 2024-08-12
# Description: This file inherits class from mmwave/dataloader/adc.py
#              and only overrides the fastRead_in_Cpp function.
#############################################

from mmwave.dataloader.adc import ADC
import numpy as np
import math
import fpga_udp

class ADC(ADC):
    # @override
    def fastRead_in_Cpp(self, numframes=1, timeOut=2, sortInC=True, isLossReturn=False):
        minPacketNum = math.ceil(self.PACKETS_IN_FRAME * numframes)
        print("min Packet Num:", minPacketNum)

        recvData = fpga_udp.read_data_udp_block_thread(
            self.data_socket.fileno(), numframes, self.BYTES_IN_FRAME, self.BYTES_OF_PACKET, timeOut, sortInC
        )
        
        if sortInC:
            recvData = np.ndarray(shape=-1, dtype=np.int16, buffer=recvData)
            receivedPacketNum = fpga_udp.get_receivedPacketNum()
            expectedPacketNum = fpga_udp.get_expectedPacketNum()
            firstPacketNum = fpga_udp.get_firstPacketNum()
            lastPacketNum = fpga_udp.get_lastPacketNum()
            print("first Packet Num:%d, last Packet Num:%d" % (firstPacketNum, lastPacketNum))
            print("received packet num:%d, expected packet num:%d, loss:%.2f%%" % (
                receivedPacketNum, expectedPacketNum, (expectedPacketNum - receivedPacketNum) / expectedPacketNum * 100
            ))
            
            if isLossReturn:
                return recvData, (expectedPacketNum - receivedPacketNum) / expectedPacketNum * 100
            else:
                return recvData
        else:
            print("All received, post processing packets...")
            recvData = np.reshape(recvData, (-1, self.BYTES_OF_PACKET))
            recvQueue = list(map(lambda x: bytes(x), recvData))
            
            receivedData, firstPacketNum, receivedPacketNum = self.postProcPacket(recvQueue, minPacketNum)
            databuf = receivedData[0:numframes * self.UINT16_IN_FRAME]
            print("received packet num:%d, expected packet num:%d, loss:%.2f%%" % (
                len(receivedPacketNum), minPacketNum, (minPacketNum - len(receivedPacketNum)) / minPacketNum * 100
            ))
            if isLossReturn:
                return databuf, (minPacketNum - len(receivedPacketNum)) / minPacketNum * 100
            else:
                return databuf
