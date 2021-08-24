# multi-class-text-classification

# data describe
text = pd.DataFrame(train['text'])
text['len'] = text['text'].str.len()
text.describe()
                 len
count  105758.000000
mean      475.768887
std       780.951261
min         6.000000
25%       238.000000
50%       345.000000
75%       514.000000
max     56181.000000

y = train['label']
c = {'label':[np.argmax(i) for i in y]}
c = pd.DataFrame(c)
c['label'].value_counts().plot(kind = 'bar')
c['label'].value_counts()
61     4840
36     4615
12     3271
1      2286
175    2123
82     1747
29     1471
43     1470
24     1457
159    1414
155    1410
53     1396
17     1375
117    1361
87     1351
3      1334
163    1329
31     1326
144    1323
112    1322
97     1322
73     1318
4      1315
85     1314
8      1309
42     1306
2      1306
80     1305
5      1303
39     1298
66     1294
52     1291
152    1289
171    1283
71     1278
33     1268
131    1266
190    1263
125    1249
156    1241
49     1239
151    1199
141    1195
88     1190
201    1147
164    1141
45     1137
187    1127
198    1126
115    1122
122    1116
180    1085
119    1043
200    1037
145     991
68      969
185     951
170     940
110     755
10      735
6       734
166     733
184     716
84      710
148     585
178     569
121     557
154     527
69      524
146     520
79      519
18      476
83      475
137     474
118     424
11      411
157     401
132     388
109     388
130     380
128     366
172     361
106     320
129     272
40      271
161     261
38      253
16      250
58      240
147     227
72      227
107     221
86      214
149     203
74      198
199     197
47      182
143     178
165     175
20      167
13      157
153     150
89      147
15      145
30      144
113     142
126     138
64      136
60      127
192     118
57      117
91      114
76      112
101     109
48      106
22      104
37       99
150      95
21       93
142      89
25       87
108      86
103      85
191      76
27       76
111      75
183      73
0        71
7        70
189      69
67       68
70       66
81       66
194      65
92       58
50       56
32       55
90       54
114      53
105      52
134      49
78       49
54       47
162      47
41       47
96       44
65       41
140      40
44       40
26       39
195      33
99       33
77       33
63       32
94       32
133      31
193      31
104      30
28       30
123      30
19       29
136      29
102      28
186      28
124      27
182      27
169      25
59       24
167      22
173      21
51       20
75       20
168      19
56       18
35       18
55       18
14       17
23       17
120      16
62       16
34       16
158      15
9        15
160      15
46       14
93       14
95       13
100      13
139      13
138      12
116      12
127       8
174       8
177       4
196       3
98        3
179       2
Name: label, dtype: int64

