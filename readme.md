# Semester project 4 - Inverted pendulum
This is a project about inverted pendulums and making it balance using both modern and classic control theory.

## Table of contents

- [About the project](#about-the-project)
- [Video showcase](#video-showcase)
- [Getting Started](#getting-started)
- [Usage](#usage)

## About the project
This project is all about control theory and balance. The goal was to balance an inverted pendulum utilizing a preexisting hardware setup with a PLC as the "brains". This poses a lot of challenges, as cycle time and variables aren't as easy to use and get working as a program like C++. Getting the data from the PLC from tests also proved quite difficult due to low storage and bad file system integration, so a OPC UA server is set up on the PLC to transfer test data. Overall 3 controllers were integrated, tested, and were working in the end, making this project a success!

## Video showcase
https://youtube.com/shorts/rq7DjSa0-gI?feature=share

## Getting started
To run tests with our code, you must first download and set things up, before running any tests.

### Prerequisites
- The hardware to run the experiment as detailed in the report
- A windows PC
- B&R Automation Studio
- Python 3.13

### Setup
1. Open the PLC program in Automation Studio
2. Set up the IP of the PLC and note it down
3. Push the program to the PLC
4. In the "dataCollection.py" change the IP to the PLC IP previously set
5. Run the Python script 
6. When the programs are actively running on the PLC and PC tests can be started

## Usage
1. When the button on the electrical cabinet is pressed a recording will start
2. Open the variable view in Automation Studio
3. Add the "ProgramState" to the view
4. Change the state to the controller to be tested