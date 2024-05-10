//https://www.geeksforgeeks.org/socket-programming-cc/
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
 
int main(int argc, char const* argv[]) {
    int port = strtol(argv[1], NULL, 10);
    int status, server_fd;
    struct sockaddr_in serv_addr;
    char* connectionMsg = "Kyle Lund: 29501039";
    char buffer[1024] = { 0 };
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }
 
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
 
    // Convert IPv4 and IPv6 addresses from text to binary
    // form
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)
        <= 0) {
        printf(
            "\nInvalid address/ Address not supported \n");
        return -1;
    }
 
    if ((status
         = connect(server_fd, (struct sockaddr*)&serv_addr,
                   sizeof(serv_addr)))
        < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }
    
    read(server_fd, buffer, 1024 - 1);
    printf("%s\n", buffer);
    send(server_fd, connectionMsg, strlen(connectionMsg), 0);
 
    // closing the connected socket
    close(server_fd);
    return 0;
}