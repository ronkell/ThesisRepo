"""digraph G
{
root [label="Y (1,3,0,0) 0.25\lA (\naction_move_down_agent1)\l" shape=doublecircle labeljust="l"];
x0row55 [label="Y (1,3,3,0) 1\lA (\naction_push_up_box1_agent2)\l"];
x0row62 [label="Y (1,3,1,0) 0.8\lA (\naction_sense_box1_agent1)\l"];
x0row70 [label="Y (1,3,1,0) 1\lA (\naction_push_left_box1_agent1)\l"];
x0row73 [label="Y (1,3,0,0) 0.8\lA (\naction_sense_box1_agent1)\l"];
x0row81 [label="Y (1,3,0,0) 1\lA (action_idle)\l"];
x0row85 [label="Y (3,3,0,0) 1\lA (action_idle)\l"];
x0row92 [label="Y (3,3,0,3) 1\lA (\naction_push_up_box2_agent2)\l"];
x0row102 [label="Y (3,3,0,1) 0.8\lA (\naction_sense_box2_agent1)\l"];
x0row105 [label="Y (3,3,0,1) 1\lA (\naction_move_up_agent1)\l"];
x0row112 [label="Y (1,3,0,1) 1\lA (\naction_push_left_box2_agent1)\l"];
x0row117 [label="Y (1,3,0,0) 0.8\lA (\naction_sense_box2_agent1)\l"];
x0row4 [label="Y (3,3,0,0) 0.25\lA (\naction_sense_box1_agent1)\l"];
x0row15 [label="Y (3,3,3,0) 0.5\lA (\naction_sense_box2_agent1)\l"];
x0row16 [label="Y (3,3,0,0) 0.5\lA (\naction_sense_box2_agent1)\l"];
x0row19 [label="Y (3,3,3,3) 1\lA (\naction_move_up_agent1)\l"];
x0row20 [label="Y (3,3,3,0) 1\lA (\naction_move_up_agent1)\l"];
x0row30 [label="Y (1,3,3,3) 1\lA (\naction_push_up_box2_agent2)\l"];
x0row39 [label="Y (1,3,3,1) 0.8\lA (\naction_sense_box2_agent1)\l"];
x0row42 [label="Y (1,3,3,1) 1\lA (\naction_push_left_box2_agent1)\l"];
x0row51 [label="Y (1,3,3,0) 0.8\lA (\naction_sense_box2_agent1)\l"];
root -> x0row4 [label="o (null,null) 1\l"];
x0row4 -> x0row15 [label="o (yes,null) 0.5\l"];
x0row4 -> x0row16 [label="o (no,null) 0.5\l"];
x0row15 -> x0row19 [label="o (null,yes) 0.5\l"];
x0row15 -> x0row20 [label="o (null,no) 0.5\l"];
x0row19 -> x0row30 [label="o (null,null) 1\l"];
x0row30 -> x0row39 [label="o (null,null) 1\l"];
x0row39 -> x0row42 [label="o (null,yes) 0.8\l"];
x0row39 -> x0row30 [label="o (null,no) 0.2\l"];
x0row42 -> x0row51 [label="o (null,null) 1\l"];
x0row51 -> x0row42 [label="o (null,yes) 0.2\l"];
x0row51 -> x0row55 [label="o (null,no) 0.8\l"];
x0row55 -> x0row62 [label="o (null,null) 1\l"];
x0row62 -> x0row70 [label="o (yes,null) 0.8\l"];
x0row62 -> x0row55 [label="o (no,null) 0.2\l"];
x0row70 -> x0row73 [label="o (null,null) 1\l"];
x0row73 -> x0row70 [label="o (yes,null) 0.2\l"];
x0row73 -> x0row81 [label="o (no,null) 0.8\l"];
x0row81 -> x0row81 [label="o (null,null) 1\l"];
x0row20 -> x0row55 [label="o (null,null) 1\l"];
x0row16 -> x0row92 [label="o (null,yes) 0.5\l"];
x0row16 -> x0row85 [label="o (null,no) 0.5\l"];
x0row92 -> x0row102 [label="o (null,null) 1\l"];
x0row102 -> x0row92 [label="o (null,yes) 0.2\l"];
x0row102 -> x0row105 [label="o (null,no) 0.8\l"];
x0row105 -> x0row112 [label="o (null,null) 1\l"];
x0row112 -> x0row117 [label="o (null,null) 1\l"];
x0row117 -> x0row112 [label="o (null,yes) 0.2\l"];
x0row117 -> x0row81 [label="o (null,no) 0.8\l"];
x0row85 -> x0row85 [label="o (null,null) 1\l"];
}
"""
# this is only for 2x2_2A_1L_1H





s={'b':[(0,1),(1,0),(1,1),(1,1)], 'action':(4,0),'next':{(0,0):'s0'},'prob':0.25}
s0={'b':[(0,0),(1,0),(1,1),(1,1)], 'action':(5,0),'next':{(1,0):'s01',(2,0):'s02'},'prob':0.25}
s01={'b':[(0,0),(1,0),(1,1),(1,1)], 'action':(6,0),'next':{(1,0):'s011',(2,0):'s012'},'prob':0.5}
s011={'b':[(0,0),(1,0),(1,1),(1,1)], 'action':(0,0),'next':{(0,0):'final'},'prob':1}
s012={'b':[(0,0),(1,0),(1,1),(0,0)], 'action':(0,1),'next':{(0,0):'s0120'},'prob':1}
s0120={'b':[(0,0),(0,0),(1,1),(0,0)], 'action':(12,12),'next':{(0,0):'s01200'},'prob':1}
s01200={'b':[(0,0),(0,0),(1,1),(1,0)], 'action':(5,0),'next':{(1,0):'s012001',(2,0):'s0120'},'prob':0.8}
s012001={'b':[(0,0),(0,0),(1,1),(1,0)], 'action':(2,0),'next':{(0,0):'s0120010'},'prob':1}
s0120010={'b':[(1,0),(0,0),(1,1),(1,0)], 'action':(0,2),'next':{(0,0):'s01200100'},'prob':1}
s01200100={'b':[(1,0),(1,0),(1,1),(1,0)], 'action':(13,13),'next':{(0,0):'s012001000'},'prob':0.8}
s012001000={'b':[(1,0),(1,0),(1,1),(1,0)], 'action':(6,0),'next':{(1,0):'s0120010001',(2,0):'s01200100'},'prob':0.8}
s0120010001={'b':[(1,0),(1,0),(1,1),(1,1)], 'action':(0,0),'next':{(0,0):'final'},'prob':1}

s02={'b':[(0,0),(1,0),(0,0),(1,1)], 'action':(6,0),'next':{(1,0):'s021',(2,0):'s022'},'prob':0.5}
s021={'b':[(0,0),(1,0),(0,0),(1,1)], 'action':(8,0),'next':{(0,0):'s0210'},'prob':1}
s0210={'b':[(0,0),(1,0),(1,0),(1,1)], 'action':(5,0),'next':{(1,0):'s02101',(2,0):'s021'},'prob':0.8}
s02101={'b':[(0,0),(1,0),(1,0),(1,1)], 'action':(0,9),'next':{(0,0):'s021010'},'prob':1}
s021010={'b':[(0,0),(1,0),(1,1),(1,1)], 'action':(0,5),'next':{(0,1):'s0210101',(0,2):'s02101'},'prob':0.8}
s0210101={'b':[(0,0),(1,0),(1,1),(1,1)], 'action':(0,0),'next':{(0,0):'final'},'prob':1}

s022={'b':[(0,0),(1,0),(0,0),(0,0)], 'action':(8,0),'next':{(0,0):'s0220'},'prob':1}
s0220={'b':[(0,0),(1,0),(1,0),(0,0)], 'action':(5,0),'next':{(1,0):'s02201',(2,0):'s022'},'prob':0.8}
s02201={'b':[(0,0),(1,0),(1,0),(0,0)], 'action':(0,9),'next':{(0,0):'s022010'},'prob':1}
s022010={'b':[(0,0),(1,0),(1,1),(0,0)], 'action':(0,5),'next':{(0,1):'s0220101',(0,2):'s02201'},'prob':0.8}
s0220101={'b':[(0,0),(1,0),(1,1),(0,0)], 'action':(0,1),'next':{(0,0):'s02201010'},'prob':1}
s02201010={'b':[(0,0),(0,0),(1,1),(0,0)], 'action':(12,12),'next':{(0,0):'s022010100'},'prob':1}
s022010100={'b':[(0,0),(0,0),(1,1),(1,0)], 'action':(6,0),'next':{(1,0):'s0220101001',(2,0):'s02201010'},'prob':0.8}
s0220101001={'b':[(0,0),(0,0),(1,1),(1,0)], 'action':(2,0),'next':{(0,0):'s02201010010'},'prob':1}
s02201010010={'b':[(1,0),(0,0),(1,1),(1,0)], 'action':(0,2),'next':{(0,0):'s022010100100'},'prob':1}
s022010100100={'b':[(1,0),(1,0),(1,1),(1,0)], 'action':(13,13),'next':{(0,0):'s0220101001000'},'prob':1}
s0220101001000={'b':[(1,0),(1,0),(1,1),(1,1)], 'action':(6,0),'next':{(1,0):'s02201010010001',(2,0):'s022010100100'},'prob':0.8}
s0220101001000={'b':[(1,0),(1,0),(1,1),(1,1)], 'action':(0,0),'next':{(0,0):'final'},'prob':1}


mymap={'s0':s0,'s01':s01,'s011':s011,'s012':s012,'s0120':s0120,'s01200':s01200,'s012001':s012001,'s0120010':s0120010,'s01200100':s01200100,
        's012001000':s012001000,'s0120010001':s0120010001,
       's02':s02,'s021':s021,'s0210':s0210,'s02101':s02101,'s021010':s021010,'s0210101':s0210101,
       's022':s022,'s0220':s0220,'s02201':s02201,'s022010':s022010,'s0220101':s0220101,'s02201010':s02201010,
       's022010100':s022010100,'s0220101001':s0220101001,'s02201010010':s02201010010,'s022010100100':s022010100100,
       's0220101001000':s0220101001000,'s0220101001000':s0220101001000}






























