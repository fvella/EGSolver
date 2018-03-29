
#ifndef FLAGLOADSIGHANDLER_H
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

extern uint timeout_expired;

void install_alarmhandler();
void install_handler();

void my_catchint(int signo);
void my_catchalarm(int signo);

#define FLAGLOADSIGHANDLER_H 1
#endif
