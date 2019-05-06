import socket

class Server(object):

    def __init__(self, host, port):

        # 소켓이란 네트워크를 통하는 컴퓨터의 외부와 컴퓨터 내부의 프로그램을 이어주는 인터페이스
        # 소켓을 생성(패밀리. 소켓), (AF_INET, SOCK_STREAM은 가장 많이 쓰임)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 서버가 특정 포트를 열고 입력을 기다리기 위해서는 소켓을 포트에 바인드 하는 과정이 선행되어야함
        # bind로 넘기는 인자는 튜플로 감싸서 전달
        self.server_socket.bind((host, port))

        # 클라이언트가 해당 포트에 접속하는 것을 기다림
        # 접속시도를 알아챘다면 서버에서도 그 요청을 받아서 접속을 시작함
        self.server_socket.listen(5)

        # 서버는 최초 생성되어 듣는 소켓이 아닌 accept()의 리턴으로 제공되는 소켓을
        # 사용하여 클라이언트와 정보를 주고 받을 수 있다.
        self.connection = self.server_socket.accept()[0]
        
    def Get_Client(self):
        # socket 객체를 넘겨줌
        return self.connection

    def __del__(self):
        self.connection.close()
        self.server_socket.close()
