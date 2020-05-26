import numpy as np
import cv2
import math
import random
from collections import deque

#CONSTANTS---------------------------------------------------------------------
COLOR_WHITE = (255,255,255)
COLOR_BLACK = (0,0,0)
COLOR_BLUE = (220,50,0)
COLOR_GREEN = (10,220,10)
COLOR_RED = (10,10,220)
MAIN_FONT = (cv2.FONT_HERSHEY_SIMPLEX,2)
SMALL_FONT = (cv2.FONT_HERSHEY_PLAIN,2)
BOLD_FONT = (cv2.FONT_HERSHEY_DUPLEX,cv2.LINE_AA)
SIZE_S = 0.5
SIZE_M = 0.75
SIZE_L = 1.0
K_PI = math.pi
K_2PI = K_PI * 2.0
K_PI_2 = K_PI / 2.0
K_3PI_2 = K_PI * 1.5

PUCK_SPEED = 40

#VARIABLES---------------------------------------------------------------------

#Game state control
gameState = 0
lSet = 0
rSet = 0
lScore = 0
rScore = 0
hand_l_hist = None
hand_r_hist = None
lastLPos = None
lastRPos = None
puckPos = None
puckAngle = 0
#angle right = 0 or 2pi,down = 1/2pi,left=pi,top = 3/2pi
puckSpeed = 0
startRoundCounter = 0
goalEffectCounter = 0.0
gameOver = False

#HELPER FUNCTIONS--------------------------------------------------------------

#Load PNG image
def loadPNG(name):
    tmp = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    return tmp

#function for drawing text onto cv2 image
def drawText(img,text,font,position,color,scale,pivot):
    text_width, text_height = cv2.getTextSize(text, font[0], scale, font[1])[0]
    displayPosition = (int(position[0] - text_width * pivot[0]),int(position[1] - text_height * pivot[1]))
    cv2.putText(img,text, displayPosition, font[0], scale, color, font[1])

#draw RGBA image on RGB image
def drawImage(srcImg,dstImg,position,pivot):
    h,w,_ = srcImg.shape
    mh,mw,_ = dstImg.shape

    x = int(position[0]) - int(w * pivot[0])
    y = int(position[1]) - int(h * pivot[1])

    #no outside draw
    # if x < -w or y < -h or x > mw or y > mh:
    #     return

    #TODO: advance draw (draw partially on dst)
    # imgX1 = 0
    # imgY1 = 0
    # imgX2 = w-1
    # imgY2 = h-1
    # if x < 0:
    #     imgX1 = -x
    #     w = w + x
    #     x = 0
    # if y < 0:
    #     h = h + y
    #     y = 0
    # if x + w > mw:
    #     w = mw - x
    # if y + h > mh:
    #     h = mh - y

    bg = dstImg[y:y+h, x:x+w]
    #Base on alpha, create black placement
    np.multiply(bg, np.atleast_3d(255 - srcImg[:, :, 3])/255.0, out=bg, casting="unsafe")
    #Add premulitply alpha into black placement
    np.add(bg, srcImg[:, :, 0:3] * (np.atleast_3d(srcImg[:, :, 3]/255) ), out=bg, casting="unsafe")
    # put the changed image back into the scene
    dstImg[y:y+h, x:x+w] = bg

#Draw rectangle of color
def drawFillRec(img,stPos,enPos,color):
    cv2.rectangle(img, stPos, enPos, color, -1)

#Prevent from out of bounds error
def validPos(centerPos,pad,rangeX,rangeY):
    ret = centerPos
    if centerPos[0] - pad[0] < rangeX[0]:
        ret[0] = rangeX[0] + pad[0]
    if centerPos[1] - pad[1] < rangeY[0]:
        ret[1] = rangeY[0] + pad[1]
    if centerPos[0] + pad[0] > rangeX[1]:
        ret[0] = rangeX[1] - pad[0]
    if centerPos[1] + pad[1] > rangeY[1]:
        ret[1] = rangeY[1] - pad[1]
    return ret

#Calculate distance
def distance(p1,p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)

#Clip angle
def clipAngle(inp):
    if inp < 0:
        return inp + K_2PI
    if inp > K_2PI:
        return inp - K_2PI
    return inp

#Check range
def inRange(val,range):
    return val >= range[0] and val <= range[1]

#Calculate angle between two points
def angle2P(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

#Light image up
def lightImage(img,gamma):
    invGamma = 1.0 / max(gamma,0.1)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

#Draw arrowed line
def drawArrowToAngle(img,center,angle,len,color):
    endP = (int(center[0] + math.cos(angle) * len),int(center[1] + math.sin(angle) * len))
    img = cv2.arrowedLine(img, center, endP, color, 4) 

#INIT--------------------------------------------------------------------------

#Setup camera
cap = cv2.VideoCapture(0)
_, img = cap.read()
IMG_H,IMG_W,_ = img.shape
IMG_SCALE = 720.0/IMG_H
IMG_H = int(720)
IMG_W = int(IMG_W * IMG_SCALE)

cv2.namedWindow( "PyHockey", cv2.WINDOW_AUTOSIZE );
#cv2.namedWindow( "Test", cv2.WINDOW_AUTOSIZE );

#Process on downscale version
DOWN_SCALE = 2.0
IMG_W2 = int(IMG_W/DOWN_SCALE)
IMG_H2 = int(IMG_H/DOWN_SCALE)
print("Original size ",img.shape)
print("Processing size (",IMG_H2,",",IMG_W2,")")

HAND_L_X = 0.4
HAND_R_X = 1.6

BORDER_SIZE = 8

GOAL_H = int(IMG_H * 0.6)
GOAL_Y1 = int((IMG_H - GOAL_H) / 2.0)
GOAL_Y2 = GOAL_Y1 + GOAL_H

#Load BGRA images
I_LOGO = loadPNG("logo.png")
I_PUCK = loadPNG("puck.png")
I_L_PUSHER = loadPNG("puck_red.png")
I_R_PUSHER = loadPNG("puck_blue.png")
I_WIN = loadPNG("win.png")
I_LOSE = loadPNG("lose.png")

PUCK_D = 96
PUCK_R = int(PUCK_D/2)
PUSHER_D = 160
PUSHER_R = int(PUSHER_D/2)

#GAME CONTROL FUNCTIONS--------------------------------------------------------

def scored(leftScored):
    global gameState, goalEffectCounter,lScore,rScore
    gameState = 3
    goalEffectCounter = 0
    if leftScored:
        lScore = lScore + 1
    else:
        rScore = rScore + 1

def resetRound():
    global gameState,startRoundCounter,puckPos,puckAngle
    gameState = 1
    startRoundCounter = 0
    puckPos = (IMG_W2,IMG_H2)
    puckAngle = random.random() * K_2PI
    puckSpeed = 0

def startRound():
    global gameState,puckSpeed
    gameState = 2
    puckSpeed = PUCK_SPEED

def resetGame():
    global gameState,lSet,rSet,hand_l_hist,hand_r_hist,lastLPos,lastRPos,puckPos,puckSpeed,gameOver
    gameState = 0
    lSet = 0
    rSet = 0
    hand_l_hist = -1
    hand_r_hist = -1
    lastLPos = None
    lastRPos = None
    puckPos = None
    puckSpeed = 0
    gameOver = False

def startGame():
    global gameState,lScore,rScore,puckPos,puckAngle,lastLPos,lastRPos,lPosQ,rPosQ,puckSpeed
    gameState = 1
    lScore = 0
    rScore = 0
    lastLPos = (int(IMG_W2/2),IMG_H2)
    lastRPos = (IMG_W2 + int(IMG_W2/2),IMG_H2)
    lPosQ = deque([[lastLPos[0],lastLPos[1]]])
    rPosQ = deque([[lastRPos[0],lastRPos[1]]])
    resetRound()

INPUT_SIZE = 10
INPUT_PER_ROW = 3
INPUT_PER_COL = 3
INPUT_COUNT = INPUT_PER_ROW * INPUT_PER_COL
def calculateHandInputPoint(centerPos):
    gapX = int((IMG_W2 / 10.0) - INPUT_SIZE * 2)
    if gapX < 30:
        gapX = 30
    gapY = int(1.5 * gapX)
    w = gapX * (INPUT_PER_ROW - 1) + INPUT_SIZE * INPUT_PER_ROW
    h = gapY * (INPUT_PER_COL - 1) + INPUT_SIZE * INPUT_PER_COL
    offsetX = centerPos[0] - int(w/2.0)
    offsetY = centerPos[1] - int(h/2.0)
    ret = []
    for i in range(INPUT_COUNT):
        xx = offsetX + (i % INPUT_PER_ROW * (INPUT_SIZE + gapX))
        yy = offsetY + (int(i / INPUT_PER_ROW) * (INPUT_SIZE + gapY))
        ret.append([xx,yy,xx+INPUT_SIZE,yy+INPUT_SIZE])
    return ret

#draw hand input
def drawHandInput(img,centerPos,color):
    squares = calculateHandInputPoint(centerPos)
    for i in range(len(squares)):
        cv2.rectangle(img, (squares[i][0], squares[i][1]), (squares[i][2], squares[i][3]), color, int(DOWN_SCALE))

MAXRANGE1 = 180
MAXRANGE2 = 256
def acceptHandInput(img,centerPos):
    #convert to HSV for easier skin color tracking
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #An HSV image strip of the sample
    strip = np.zeros([INPUT_SIZE,INPUT_COUNT*INPUT_SIZE,3], dtype=hsv.dtype)

    squares = calculateHandInputPoint(centerPos)
    #stitch images
    for i in range(len(squares)):
        strip[0:INPUT_SIZE,i*INPUT_SIZE:i*INPUT_SIZE+INPUT_SIZE] = hsv[squares[i][1]:squares[i][3], squares[i][0]:squares[i][2]]
    #calculate histogram
    calHist = cv2.calcHist([strip],[0, 1], None, [MAXRANGE1, MAXRANGE2], [0, MAXRANGE1, 0, MAXRANGE2])
    return calHist

DOT_SIZE = 15
MIN_INTENSITIY = 90
def calculateCentroid(img, hist): 
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Calculate back from histogram
    calImg = cv2.calcBackProject([hsvImg], [0,1], hist, [0,MAXRANGE1,0,MAXRANGE2], 1)

    #Filter all light pixel with a white circle
    filter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DOT_SIZE,DOT_SIZE))
    cv2.filter2D(calImg, -1, filter, calImg)
    #Remove all small values to prevent wrong detections
    _,calImg = cv2.threshold(calImg, MIN_INTENSITIY, 255, cv2.THRESH_BINARY)

    #Find centroid of the biggest contour
    contours,_ = cv2.findContours(calImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None and len(contours) > 0:
        maxC = max(contours, key = cv2.contourArea)
        #Calculate centroid of contour
        m = cv2.moments(maxC)
        if m['m00'] != 0:
            #Not divide by 0
            return (int(m['m10']/m['m00']),int(m['m01']/m['m00']))

    return None

def calculatePusherP(img, hist, lastPos, offsetX, posQ):
    retP = lastPos
    p = lastPos
    p_ = calculateCentroid(img,hist)
    if p_ is not None:
        p = [element * DOWN_SCALE for element in p_]
        p[0] = p[0] + offsetX
        p = validPos(p,(PUSHER_R,PUSHER_R),(offsetX + BORDER_SIZE,offsetX + IMG_W2 - BORDER_SIZE),(BORDER_SIZE,IMG_H - BORDER_SIZE))

    #Make sure the pusher moves gradually, not instant
    if lastPos is not None:
        d = distance(lastPos,p)
        if d<200:
            retP = (lastPos[0] + (p[0] - lastPos[0])/2, lastPos[1] + (p[1] - lastPos[1])/2)
    else:
        retP = p

    #Smooth movement
    posQ.append([retP[0],retP[1]])
    if len(posQ) > 3:
        posQ.popleft()
    listQ = np.array(posQ)
    retP = (np.average(listQ[:,0]),np.average(listQ[:,1]))

    return retP


#MAIN LOOP---------------------------------------------------------------------

while 1:
    #Record keyboard input
    keyPressed = cv2.waitKey(1) & 0xFF
    #Exit
    if keyPressed == ord('q') or keyPressed == 27:
        break

    #Prepare images
    ret, img = cap.read()
    img = cv2.flip(img , 1)
    img = cv2.resize(img,(IMG_W,IMG_H))
    #(smaller) image for faster processing
    img_p = cv2.resize(img,(IMG_W2,IMG_H2))
    img_p_l = img_p[0:IMG_H2,:int(IMG_W2/2)]
    img_p_r = img_p[0:IMG_H2,int(IMG_W2/2):]

    #Update logic

    #User action to set left hand
    if gameState == 0 and lSet == 0 and keyPressed == ord('z'):
        print("SAVE LEFT HAND COLOR")
        lSet = 1
        hand_l_hist = acceptHandInput(img,(int(IMG_W2 * HAND_L_X),IMG_H2))
        if rSet == 1:
            startGame()
    #User action to set right hand
    if gameState == 0 and rSet == 0 and keyPressed == ord('m'):
        print("SAVE RIGHT HAND COLOR")
        rSet = 1
        hand_r_hist = acceptHandInput(img,(int(IMG_W2 * HAND_R_X),IMG_H2))
        if lSet == 1:
            startGame()
    #User action to reset Game
    if gameState != 0 and keyPressed == ord('g'):
        print("RESET GAME")
        resetGame()

    #Translate hand input to pusher position
    if gameState != 0 and not gameOver:
        #Calculate left hand pos
        if lSet:
            lastLPos = calculatePusherP(img_p_l,hand_l_hist,lastLPos,0,lPosQ)
        #Calculate right hand pos
        if rSet:
            lastRPos = calculatePusherP(img_p_r,hand_r_hist,lastRPos,IMG_W2,rPosQ)

    #Main GamePlay
    if gameState == 2:
        PUSHER_PUCK = PUSHER_R + PUCK_R
        dToLPusher = distance(lastLPos,puckPos)
        if dToLPusher <= PUSHER_PUCK:
            puckAngle = angle2P(lastLPos,puckPos)
            puckPos = (lastLPos[0] + math.cos(puckAngle) * PUSHER_PUCK,lastLPos[1] + math.sin(puckAngle) * PUSHER_PUCK)
        dToRPusher = distance(lastRPos,puckPos)
        if dToRPusher <= PUSHER_PUCK:
            puckAngle = angle2P(lastRPos,puckPos)
            puckPos = (lastRPos[0] + math.cos(puckAngle) * PUSHER_PUCK,lastRPos[1] + math.sin(puckAngle) * PUSHER_PUCK)
        #Calculate puck position
        puckX = puckPos[0] + math.cos(puckAngle) * puckSpeed
        puckY = puckPos[1] + math.sin(puckAngle) * puckSpeed
        #Collide board border LEFT
        if puckX - PUCK_R < BORDER_SIZE:
            #Scoring
            if inRange(puckY,(GOAL_Y1,GOAL_Y2)):
                scored(False)
            else:
                puckX = BORDER_SIZE + PUCK_R
                if inRange(puckAngle,(K_PI_2,K_PI)):
                    puckAngle = K_PI_2 - (puckAngle - K_PI_2)
                elif inRange(puckAngle,(K_PI,K_3PI_2)):
                    puckAngle = K_3PI_2 + (K_3PI_2 - puckAngle)
        #RIGHT
        if puckX + PUCK_R > IMG_W - BORDER_SIZE:
            #Scoring
            if inRange(puckY,(GOAL_Y1,GOAL_Y2)):
                scored(True)
            else:
                puckX = IMG_W - BORDER_SIZE - PUCK_R
                if inRange(puckAngle,(0,K_PI_2)):
                    puckAngle = K_PI - puckAngle
                elif inRange(puckAngle,(K_3PI_2,K_2PI)):
                    puckAngle = K_3PI_2 - (puckAngle - K_3PI_2)
        #TOP
        if puckY - PUCK_R < BORDER_SIZE:
            puckY = BORDER_SIZE + PUCK_R
            if inRange(puckAngle,(K_PI,K_2PI)):
                puckAngle = K_2PI - puckAngle
        #BOTTOM
        if puckY + PUCK_R > IMG_H - BORDER_SIZE:
            puckY = IMG_H - BORDER_SIZE - PUCK_R
            if inRange(puckAngle,(0,K_PI)):
                puckAngle = K_2PI - puckAngle

        puckAngle = clipAngle(puckAngle)
        puckPos = (puckX,puckY)

    elif gameState == 3:
        goalEffectCounter = goalEffectCounter + 0.1
        if lScore == 5 or rScore == 5:
            if goalEffectCounter > 3:
                goalEffectCounter = 0
            gameOver = True
        else:
            if goalEffectCounter > 3:
                resetRound()

    elif gameState == 1:
        startRoundCounter = startRoundCounter + 1
        if startRoundCounter > 60:
            startRound()

    #Draw logic on img
    if gameState == 0:
        if lSet == 0:
            drawHandInput(img,(int(IMG_W2 * HAND_L_X),IMG_H2),COLOR_RED)
            drawText(img,'Player1: Press [z] to save hand colors.',SMALL_FONT,(int(IMG_W2 * 0.4),IMG_H - 80),COLOR_RED,SIZE_L,(0.5,0.5))
        if rSet == 0:
            drawHandInput(img,(int(IMG_W2 * HAND_R_X),IMG_H2),COLOR_BLUE)
            drawText(img,'Player2: Press [m] to save hand colors.',SMALL_FONT,(int(IMG_W2 * 1.6),IMG_H - 80),COLOR_BLUE,SIZE_L,(0.5,0.5))
        
        drawImage(I_LOGO,img,(IMG_W2,IMG_H2),(0.5,0.5))
        drawText(img,'Press [q] or [esc] to exit.',MAIN_FONT,(IMG_W2,IMG_H - 10),COLOR_GREEN,SIZE_L,(0.5,1.0))
    elif gameState == 1 or gameState == 2 or gameState == 3:
        #Game border
        cv2.rectangle(img, (0,0), (IMG_W,IMG_H), COLOR_BLACK, BORDER_SIZE*2)

        #goals
        drawFillRec(img,(0,GOAL_Y1),(BORDER_SIZE,GOAL_Y2),COLOR_RED)
        drawFillRec(img,(IMG_W - BORDER_SIZE,GOAL_Y1),(IMG_W,GOAL_Y2),COLOR_BLUE)

        if gameState != 3:
            drawText(img,str(lScore),MAIN_FONT,(20,40),COLOR_RED,SIZE_L,(0.0,0.0))
            drawText(img,str(rScore),MAIN_FONT,(IMG_W - 20,40),COLOR_BLUE,SIZE_L,(1.0,0.0))

        drawImage(I_L_PUSHER,img,lastLPos,(0.5,0.5))
        drawImage(I_R_PUSHER,img,lastRPos,(0.5,0.5))

        if gameState == 1:
            if startRoundCounter < 50 :
                drawArrowToAngle(img,puckPos,puckAngle -(50 - startRoundCounter)/20.0,PUCK_D,COLOR_GREEN)
            else:
                drawArrowToAngle(img,puckPos,puckAngle,PUCK_D,COLOR_GREEN)

            if lScore == 0 and rScore == 0:
                drawText(img,'First to 5 wins!',MAIN_FONT,(IMG_W2,IMG_H - 50),COLOR_GREEN,SIZE_L,(0.5,1.0))
        #Draw puck when not scoring
        if gameState != 3:
            drawImage(I_PUCK,img,puckPos,(0.5,0.5))

        #Scoring effect
        if gameState == 3:
            img = lightImage(img,goalEffectCounter)
            drawText(img,str(lScore),BOLD_FONT,(IMG_W2*HAND_L_X,IMG_H2),COLOR_RED,10.0,(0.5,0.0))
            drawText(img,str(rScore),BOLD_FONT,(IMG_W2*HAND_R_X,IMG_H2),COLOR_BLUE,10.0,(0.5,0.0))
            if gameOver:
                if lScore == 5:
                    drawImage(I_WIN,img,(IMG_W2*HAND_L_X,IMG_H * 0.7),(0.5,0.5))
                    drawImage(I_LOSE,img,(IMG_W2*HAND_R_X,IMG_H * 0.7),(0.5,0.5))
                else:
                    drawImage(I_LOSE,img,(IMG_W2*HAND_L_X,IMG_H * 0.7),(0.5,0.5))
                    drawImage(I_WIN,img,(IMG_W2*HAND_R_X,IMG_H * 0.7),(0.5,0.5))

        #main texts
        drawText(img,'Press [g] to reset.',SMALL_FONT,(IMG_W2,IMG_H - 10),COLOR_GREEN,SIZE_L,(0.5,1.0))


    #Show the image on a window
    cv2.imshow('PyHockey',img)
#end main loop

#CONCLUDE SESSION--------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()