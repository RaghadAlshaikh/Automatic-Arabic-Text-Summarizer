# -- coding: utf-8 --
"""
Created on Thu Mar 12 02:15:15 2020

@author: Raghad Alshaikh, Ghaidaa Aflah, Nada Alamouadi
"""
#import nltk 
import heapq
import math

# ============================ 
# ====== Load input ======
# it is Documents 
documents=["ديناصور حيوان فقاري ساد في النظام البيئي الأرضي لأكثر من 160 مليون سنة .",
      "ول الديناصورات ظهر قبل حوالي 230 مليون سنة خلت أما آخر الديناصورات على ظهر الأرض فاختفت في حادثة انقراض كارثية ، في نهاية العصر الكريتاسي .",
      " قبل 65 مليون سنة . ",
      "يعتبر الخبراء الآن الطيور الجديثة الأحفاد المباشرين المتحدرين من الديناصورات الثيروبودية .",
      "منذ أن تم وصف الديناصور للمرة الأولى في القرن التاسع عشر لقيت هياكل الديناصورات المستحاثية اهتماما واسعا من المتحف على امتداد العالم .",
      " أصبح الديناصور جزءا من ثقافة العالم و اكتسب شعبية واسعة منذ ذلك الحين ، بالذات بين الأطفال . ",
      "و كثيرا ما استخدم في الكتب الأكثر مبيعا و في أفلام الخيال العلمي و أهمها الحديقة الجوراسية.",
     "في الاستخدام غير الرسمي غير العلمي يتم استخدام مصطلح ديناصور من اجل الإشارة إلى كل زاحف قبل تاريخي ، مثل بيليكوسور , ديميترودون ، و البتيروسور المجنح ، و إشثيوسور المائي ، و بليسيوسور و موساسور ، مع أن جميع هذه الكائنات عمليا و علميا ليست ديناصورات ." ]

#=============================
# ============================ 
# ====== Load input ======
allparagraphContent_Cleaned = "لا يوجد اسم ثانِ يُعرف به القمر غير القمر، بيد أن كلمة قمر تستخدم للإشارة إلى أي جرم سماوي أو صناعي، يقوم بمدار معين حول الأرض، أو أيِ من الكواكب الأخرى، فكوكب زحل مثلاً له ثمانية عشر قمراً تابعاً. وهناك تسمية أُخرى للقمر ويسمّى بها أحياناً وهي لونا. وجانب القمر الذي لا يُرى من الأرض يسمّى الجانب البعيد، أو الجانب المظلم، وسمّي بهذا الإسم لعدم قدرة بني البشر من النظر إليه من الأرض، فلو كانت هناك مركبة فضائية على هذا الجانب المظلم فسيتعذر الإتصال اللاسلكي بين الأرض وبين مركبة الفضاء. سيتركز هذا المقال عن القمر المتعارف عليه بين الناس، وهو القمر التابع للأرض.يقوم القمر بدورة كاملة حول الأرض مرة واحدة كل 4 أسابيع تقريباً، وفي كلّ ساعة تمر، يتحرك القمر بمقدار نصف درجة، ويمضي القمر في مدار له يميل على دائرة البروج بنحو 5 درجات.تقاس دورة القمر حول الأرض بالأشهر النجمية وبالأشهر الأقترانية.الدّورة النجمية وهي الفترة الزمنية التي يحتاجها القمر ليدور دورة واحدة حول الأرض بالنسبة للنجوم، وتستغرق 27 يوماً وثلث اليوم. الدّورة الأقترانية وهي الفترة الزمنية التي يحتاجها القمر ليدور دورة واحدة حول الأرض بالنسبة للشمس، وتستغرق 29 يوماً ونصف اليوم. وهي نفس الفترة التي يحتاجها القمر ليدور حول نفسه دورة كاملة، ولهذا السبب يرى الناظر من الأرض نفس الوجه للقمر.نتيجة تطابق الفترة الزمنية التي يأخذها القمر في دورانه حول نفسه وتلك التي يأخذها في دورانه حول الأرض، يجد أهل الأرض أن نفس الجانب من القمر مقابل للأرض ولا يتغيّر هذا الجانب. وتأثر حركة القمر بدورانه حول الأرض على بحار ومحيطات الأرض وتسبب ظاهرة المد والجزر التي نعرفها. وقد إختلف العلماء على مرّ السنين في أصل القمر وكيف آلت به الأمور على ما هو عليه، ومن أكثر النظريات التي تلقى تأييداً في الأوساط الفلكية، تلك التي تنادي بأن الأرض البكر التي نحن عليها قد إرتطم بها جسم كبير يقدّر حجمه بحجم كوكب المريخ وأقتطع هذا الجسم من الأرض ما اقتطع، وتناثر من الأرض قطع التحمت مع بعضها البعض وكوّنت القمر الذي نعرفه اليوم، وتعرف هذه النظرية بنظرية الصدمة الكبرى. وقد عمل العلماء على محاكاة نظرية الصدمة الكبرى في اغسطس من العام 2001 ونشرت المحاكاة في هذا الموقع. ولعلّ تشابه المواد المكوّنة لكتلة القمر، بتلك المعادن الموجودة على كوكب الأرض جعلت نظرية الصدمة الكبرى نظرية مقبولة في الأوساط العلمية.منذ أربع مليارات سنة ونصف، كان القمر مغطّى بالحمم البركانية المنصهرة والتي شكّلت محيطات من الحمم على سطح القمر. وتتكون قشرة القمر من المواد الأوّلية التّالية : يورانيوم، ثوريوم، بوتاسيوم، اكسجين، سيليكون، مغنيسيوم، حديد، تيتانيوم، كالسيوم، المنيوم، والهيدروجين. وعندما تسقط الإشعاعات الكونية على تلك العناصر الأولية، تقوم تلك العناصر على إنعكاس تلك الإشعاعات بخواصّ مختلفة تعتمد على طبيعة العنصر الأولي العاكس للإشعاع وبصورة إشعاعات جاما. وتجدر الإشارة ان بعض العناصر الأولية على سطح القمر تُصدر إشعاعات جاما بدون الحاجة لتعرّض تلك المواد الأولية لأي نوع من الإشعاعات الكونية كاليورانيوم أو البوتاسيوم والثوريوم.قامت النيازك والشهب بالإرتطام بالقمر مرات ومرات عديدة، ويُرى ذلك جلياُ في النتوءات الواضحة على سطح القمر. وقد حمل الكثير من تلك النيازك والشهب الماء، وحطّ على سطح القمر بمعيّة النيازك والشهب، وبمجرّد تعرض ماء النيازك والشهب لحرارة الشمس، يتفكك الماء لمكوّناته الأصلية (هيدروجين وأكسجين)، وتبدأ هذه العناصر في التطاير في الفضاء، وتبقى فرضية وجود الماء قائمة إمّا بوجوده على السطح، أو تحت قشرة القمر، وتقدّر كمية الماء على القمر ببليون متر مكعّب.يخسف القمر إذا وقعت الأرض بين أشعّة الشمس وبين جزء من القمر أو كلّ القمر، فظلّ الأرض حين تمرّ في مجراها حول الشمس يقع على القمر ويرى أهل الأرض وكأن القمر قد أُقتُطِع من نوره شيء. وننوه هنا أن ليس للقمر نور طبيعي وما النور السّاطع من القمر إلا إنعكاس أشعّة الشمس من على القمر إلى الأرض، فيراه من على الأرض وكأنه ذو نور ساطع. ولا تحدث ظاهرة خسوف القمر إلا في حالة القمر المكتمل (بدر). أمّا في ما يخصّ الكسوف، فيحصل الكسوف للشمس حين يحجب القمر أشعّة الشمس عن الأرض، وتحدث ظاهرة الكسوف في بداية تكوين القمر (هلال).أول من قام بإستكشاف الجانب المظلم من القمر كانت المركبة الفضائية السوفييتية لونا 2 عندما قامت بجولات مدارية حول القمر في 15 سبتمبر 1959، وأول من حطّ قدمه على سطح القمر هو نيل ارمسترونج، قائد المركبة الفضائية الأمريكية أبولو 11 في 20 يوليو 1969. وفي تلك الفترة، كانت الحرب الباردة في أوجها بين الإتحاد السوفييتي والولايات المتحدة، وأجّج هذا الإنجاز الأمريكي السباق إلى الفضاء بين الإتحاد السوفييتي والولايات المتّحدة. وقد وضع رائد الفضاء نيل أرمسترونج لوحة معدنية على سطح القمر كُتب فيها هنا حطّت أقدام رجال من كوكب الأرض في يوليو 1969 بعد الميلاد، لقد جئنا بسلام باسم البشرية، وقام رواد الفضاء الثلاثة بالتوقيع على اللوحة المعدنية كما وقّعها الرئيس الأمريكي آنذاك، ريتشارد نيكسون."
# ============================     
# ====== PreProcessing: Sentence Splitting====== 
sentences_tokens = allparagraphContent_Cleaned.split(".")
documents = [];
for sen in sentences_tokens:
    if len(sen)>2:
        documents.append(sen)
        
# ============================     
# ====== PreProcessing: Tokenization====== 
dictOfWords = {}
for index, sentence in enumerate(documents):
    tokenizedWords = sentence.split(' ')
    dictOfWords[index] = [(word,tokenizedWords.count(word)) for word in tokenizedWords]

# ============================     
# ====== Calculate term frequency (TF)====== 
termFrequency = {}
for i in range(0, len(documents)):
    listOfNoDuplicates = []
    for wordFreq in dictOfWords[i]:
        if wordFreq not in listOfNoDuplicates:
            listOfNoDuplicates.append(wordFreq)
        termFrequency[i] = listOfNoDuplicates

# =============================================================================
#P.S: The result were better without StopWords Removal and Arabic Stemming
#=====================remove stopwords=============================     
#stopwords_list = stopwords.words('arabic')
#print(stopwords_list)
#=====================Arabic Stemming=============================     
#st = ISRIStemmer()
#words_stemm = [st.stem(word) for word in words_tokens]
##print(words_stemm)
# ============================================================================

# ====== Normalize====== 
#To avoid problem longer (sentences)documents 
#Third: normalized term frequency       
normalizedTermFrequency = {}
for i in range(0, len(documents)):
    sentence = dictOfWords[i]
    lenOfSentence = len(sentence)
#    print(lenOfSentence)
    listOfNormalized = []
    for wordFreq in termFrequency[i]:
        normalizedFreq = wordFreq[1]/lenOfSentence
        listOfNormalized.append((wordFreq[0],normalizedFreq))
    normalizedTermFrequency[i] = listOfNormalized

# ============================     
# ====== Calculate Inverse Term Frequency (IDF)====== 
#First: put all sentences together and tokenze words
# ============================     
# ====== PreProcessing: Sentence Splitting and Tokenization====== 
allDocuments = ''
for sentence in documents:
    allDocuments += sentence + ' '
allDocumentsTokenized = allDocuments.split(' ')
# ============================   
#---Calculate IDF
allDocumentsNoDuplicates = []
for word in allDocumentsTokenized:
    if word not in allDocumentsNoDuplicates:
        allDocumentsNoDuplicates.append(word)
# ============================   
#Second calculate the number of documents where the term t appears
dictOfNumberOfDocumentsWithTermInside = {}
# ovc = vocabilary OR word
for index, voc in enumerate(allDocumentsNoDuplicates):
    count = 0
    for sentence in documents:
        if voc in sentence:
            count += 1
    dictOfNumberOfDocumentsWithTermInside[index] = (voc, count)

# ============================   
#calculate IDF
dictOFIDFNoDuplicates = {} 

for i in range(0, len(normalizedTermFrequency)):
    listOfIDFCalcs = []
    for word in normalizedTermFrequency[i]:
        for x in range(0, len(dictOfNumberOfDocumentsWithTermInside)):
            if word[0] == dictOfNumberOfDocumentsWithTermInside[x][0]:
                listOfIDFCalcs.append((word[0],math.log(len(documents)/dictOfNumberOfDocumentsWithTermInside[x][1])))
    dictOFIDFNoDuplicates[i] = listOfIDFCalcs

# ============================   
#---------Multiply tf by idf for tf-idf

dictOFTF_IDF = {}
for i in range(0,len(normalizedTermFrequency)):
    listOFTF_IDF = []
    TFsentence = normalizedTermFrequency[i]
    IDFsentence = dictOFIDFNoDuplicates[i]
    for x in range(0, len(TFsentence)):
#        print(TFsentence[x][0])
#        print(TFsentence[x][1])
        listOFTF_IDF.append((TFsentence[x][0],TFsentence[x][1]*IDFsentence[x][1]))
    dictOFTF_IDF[i] = listOFTF_IDF


# ============================     
# ====== Print Summary====== 
summary = heapq.nlargest(14, dictOFTF_IDF, key=dictOFTF_IDF.get)
print("summary")
for s in summary:
    print(documents[s])