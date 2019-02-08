from utils import Utils, OverallResultRow
import numpy
from nltk.tokenize import RegexpTokenizer
import xlwt
import datetime

if __name__ == '__main__':

    threshold = 0.70

    comparison_table_data = []
    #comparison_table_data.append(OverallResultRow(app_id, app_sentences, app_description, permission_title, app_tag))
    comparison_table_data.append(OverallResultRow("Permission", "SI", "TP", "FP", "FN", "TN", "P(%)", "R(%)", "FS(%)", "Acc(%)"))
    comparison_table_data.append(OverallResultRow("READ CONTACTS", "204", "186", "18", "49", "2930", "91.2", "79.1", "84.7", "97.9"))
    comparison_table_data.append(OverallResultRow("READ CALENDAR", "288", "241", "47", "42", "2422", "83.7", "85.1", "84.4", "96.8"))
    comparison_table_data.append(OverallResultRow("RECORD AUDIO", "259", "195", "64", "50", "3470", "75.9", "79.7", "77.4", "97.0"))
    comparison_table_data.append(OverallResultRow("TOTAL", "751", "622", "129", "141", "9061", "82.8", "81.5", "82.2", "97.3"))


    inputFolder = "IO_Input"
    outputFolder = "IO_Result"
    fileStamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # vectorFile = "IO_Input/vectors_wiki.normalized.txt"
    vectorFile = "IO_Input/wiki-news-300d-1M-subword.vec"
    # vectorFile = "IO_Input/crawl-300d-2M-subword.vec"
    # wiki_vector = ""
    # wiki_vector = Utils.read_word_vec(vectorFile)
    wiki_vector = Utils.read_word_vec(vectorFile, 1)

    chunk_gram = r"""Chunk: {<RB.?>*<VB|VBD|VBG|VBN|VBP|VBZ.>*<NNP>+<NN>?}"""

    #chunk_gram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

    # chunk_gram = r"""
    #   NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    #   PP: {<IN><NP>}               # Chunk prepositions followed by NP
    #   VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
    #   CLAUSE: {<NP><VP>}           # Chunk NP, VP
    #   CHK: {<NP><NP>}
    #   """

    # chunk_gram = r"""
    #  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    #  PP: {<IN><NP>}               # Chunk prepositions followed by NP
    #  Chunk: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
    #  Chunk: {<NP><VP>}           # Chunk NP, VP
    #  Chunk: {<NP><NP>}
    #  """
    # chunk_gram = r"""
    #  Chunk: {<DT>? <JJ>*<NN|NNS>+}     # NP: {<DT>? <JJ>*<NN|NNS>+}
    #  """

    logFileName = outputFolder + "/" + fileStamp + "_logs.txt"
    logFile = open(logFileName, "w")
    logFile.write("Vector File   -----------------------------------\n")
    logFile.write(vectorFile)
    logFile.write("\n")
    logFile.write("Chunk Gram   -----------------------------------\n")
    logFile.write(chunk_gram)
    logFile.write("\n")
    logFile.write("Threshold   -----------------------------------\n")
    logFile.write(str(threshold))
    logFile.write("\n")
    logFile.close()

    outputFileName = outputFolder + "/" + fileStamp + "_results.xls"
    outputFile = xlwt.Workbook()

    whyper_result_files = ["Read_Calendar", "Read_Contacts", "Record_Audio"]

    for inputFileName in whyper_result_files:

        words, w2i, permissions, applications = Utils.read_whyper_data(inputFolder, inputFileName +".xls", "excel", True, chunk_gram, False)

        sheet1 = outputFile.add_sheet(inputFileName)

        cols = ["Sentence", "Manually-marked", "Keyword-based", "Whyper", "MostSimilarChunk", "Similarity", "True Positive", "False Positive", "False Negative", "True Negative", "True Positive (Whypera Gore)", "False Positive (Whypera Gore)", "False Negative (Whypera Gore)", "True Negative (Whypera Gore)"]
        txt = "Row %s, Col %s"
        row = sheet1.row(0)
        for index, col in enumerate(cols):
            row.write(index, col)
        outputFileLineIndex = 1

        permissionWords = inputFileName.split("_")
        permissionWord1 = wiki_vector.get(permissionWords[0].lower())
        permissionWord2 = wiki_vector.get(permissionWords[1].lower())
        permissionVector = numpy.array(permissionWord1) + numpy.array(permissionWord2)

        for current_app in applications:

            for current_dsc_sentence in current_app.dsc_sentences:
                most_similar_result = 0
                most_similar_chunk = ""

                current_chunk_list = current_dsc_sentence.chunk_list
                chunk_total_vector = 0  # initialize
                for x in current_chunk_list:
                    tokenizer = RegexpTokenizer(r'\w+')
                    exists = 0
                    for w in tokenizer.tokenize(x):
                        if Utils.to_lower(w, True) in wiki_vector:

                            if exists == 0:
                                chunk_total_vector = wiki_vector.get(Utils.to_lower(w, True))
                                exists = 1
                            else:
                                current_chunk_vec = wiki_vector.get(Utils.to_lower(w, True))
                                chunk_total_vector = numpy.array(chunk_total_vector) + numpy.array(current_chunk_vec)
                    if exists > 0:
                        result = Utils.cos_similiariy(permissionVector, chunk_total_vector)

                        print("-----***********-----------")
                        print(x)
                        print(result)
                        print("-----***********-----------")

                        if result > most_similar_result:
                            most_similar_result = result
                            most_similar_chunk = x

                # cols = ["Sentence", "Manually-marked", "Keyword-based", "Whyper", "MostSimilarChunk", "Similarity"]
                row = sheet1.row(outputFileLineIndex)
                row.write(0, current_dsc_sentence.sentence)
                row.write(1, current_dsc_sentence.manual_marked)
                row.write(2, current_dsc_sentence.key_based)
                row.write(3, current_dsc_sentence.whyper_tool)
                row.write(4, most_similar_chunk)
                row.write(5, most_similar_result)

                # True Positive
                if (current_dsc_sentence.manual_marked == 1) & (most_similar_result >= threshold):
                    row.write(6, 1)
                # False Positive
                elif (current_dsc_sentence.manual_marked == 0) & (most_similar_result >= threshold):
                    row.write(7, 1)
                # False Negative
                elif (current_dsc_sentence.manual_marked == 1) & (most_similar_result < threshold):
                    row.write(8, 1)
                # True Negative
                elif (current_dsc_sentence.manual_marked == 0) & (most_similar_result < threshold):
                    row.write(9, 1)
                else:
                    row.write(14, "XXX")

                # True Positive (comparing with whyper)
                if (current_dsc_sentence.whyper_tool == 1) & (most_similar_result >= threshold):
                    row.write(10, 1)
                # False Positive (comparing with whyper)
                if (current_dsc_sentence.whyper_tool == 0) & (most_similar_result >= threshold):
                    row.write(11, 1)
                # False Negative (comparing with whyper)
                if (current_dsc_sentence.whyper_tool == 1) & (most_similar_result < threshold):
                    row.write(12, 1)
                # True Negative (comparing with whyper)
                if (current_dsc_sentence.whyper_tool == 0) & (most_similar_result < threshold):
                    row.write(13, 1)

                outputFileLineIndex = outputFileLineIndex + 1

        row = sheet1.row(outputFileLineIndex+2)
        row.write(6, xlwt.Formula("SUM(G2:G"+str(outputFileLineIndex)+")"))
        row.write(7, xlwt.Formula("SUM(H2:H"+str(outputFileLineIndex)+")"))
        row.write(8, xlwt.Formula("SUM(I2:I"+str(outputFileLineIndex)+")"))
        row.write(9, xlwt.Formula("SUM(J2:J"+str(outputFileLineIndex)+")"))

        row.write(10, xlwt.Formula("SUM(K2:K"+str(outputFileLineIndex)+")"))
        row.write(11, xlwt.Formula("SUM(L2:L"+str(outputFileLineIndex)+")"))
        row.write(12, xlwt.Formula("SUM(M2:M"+str(outputFileLineIndex)+")"))
        row.write(13, xlwt.Formula("SUM(N2:N"+str(outputFileLineIndex)+")"))

        row = sheet1.row(outputFileLineIndex + 4)

        row.write(6, "Precision")
        row.write(7, "Recall")
        row.write(8, "F-Score")
        row.write(9, "Accuracy")

        row.write(10, "Precision (Whypera Gore)")
        row.write(11, "Recall (Whypera Gore)")
        row.write(12, "F-Score (Whypera Gore)")
        row.write(13, "Accuracy (Whypera Gore)")

        TP = "G"+str(outputFileLineIndex+3)
        FP = "H"+str(outputFileLineIndex+3)
        FN = "I"+str(outputFileLineIndex+3)
        TN = "J"+str(outputFileLineIndex+3)

        # write table row for current permission
        row = sheet1.row(outputFileLineIndex + 5)

        # Precision --- TP / (TP + FP)
        row.write(6, xlwt.Formula(TP+"/SUM("+TP+","+FP+")"))

        # Recall --- TP / (TP + FN)
        row.write(7, xlwt.Formula(TP+"/SUM("+TP+","+FN+")"))

        # F-Score --- 2 x Precision x Recall / (Precision+Recall)
        row.write(8, xlwt.Formula("PRODUCT(2,G"+str(outputFileLineIndex+6)+",H"+str(outputFileLineIndex+6)+") / SUM(G"+str(outputFileLineIndex+6)+",H"+str(outputFileLineIndex+6)+")"))

        # Accuracy --- row.write(9, xlwt.Formula( (TP + TN) / (TP + FP + TN + FN )
        row.write(9, xlwt.Formula("SUM("+TP+","+TN+")/SUM( "+TP+","+FP+","+TN+","+FN+")"))

        TP = "K"+str(outputFileLineIndex+3)
        FP = "L"+str(outputFileLineIndex+3)
        FN = "M"+str(outputFileLineIndex+3)
        TN = "N"+str(outputFileLineIndex+3)

        # write table row for current permission
        row = sheet1.row(outputFileLineIndex + 5)

        # Precision --- TP / (TP + FP)
        row.write(10, xlwt.Formula(TP+"/SUM("+TP+","+FP+")"))

        # Recall --- TP / (TP + FN)
        row.write(11, xlwt.Formula(TP+"/SUM("+TP+","+FN+")"))

        # F-Score --- 2 x Precision x Recall / (Precision+Recall)
        row.write(12, xlwt.Formula("PRODUCT(2,K"+str(outputFileLineIndex+6)+",L"+str(outputFileLineIndex+6)+") / SUM(K"+str(outputFileLineIndex+6)+",L"+str(outputFileLineIndex+6)+")"))

        # Accuracy --- row.write(9, xlwt.Formula( (TP + TN) / (TP + FP + TN + FN )
        row.write(13, xlwt.Formula("SUM("+TP+","+TN+")/SUM( "+TP+","+FP+","+TN+","+FN+")"))

    if (inputFileName != "Read_Calendar") & (1 == 0):
        words, w2i, permissions, applications = Utils.read_whyper_data("Read_Calendar.xls", "excel", True, chunk_gram, False)

        sheetCalendar = outputFile.add_sheet("Read_Calendar")

        cols = ["Sentence", "MostSimilarChunk", "Similarity"]
        txt = "Row %s, Col %s"
        row = sheetCalendar.row(0)
        for index, col in enumerate(cols):
            row.write(index, col)
        outputFileLineIndex = 1

        for current_app in applications:

            for current_dsc_sentence in current_app.dsc_sentences:
                most_similar_result = 0
                most_similar_chunk = ""

                current_chunk_list = current_dsc_sentence.chunk_list
                chunk_total_vector = 0  # initialize
                for x in current_chunk_list:
                    tokenizer = RegexpTokenizer(r'\w+')
                    exists = 0
                    for w in tokenizer.tokenize(x):
                        if Utils.to_lower(w, True) in wiki_vector:

                            if exists == 0:
                                chunk_total_vector = wiki_vector.get(Utils.to_lower(w, True))
                                exists = 1
                            else:
                                current_chunk_vec = wiki_vector.get(Utils.to_lower(w, True))
                                chunk_total_vector = numpy.array(chunk_total_vector) + numpy.array(current_chunk_vec)
                    if exists > 0:
                        result = Utils.cos_similiariy(permissionVector, chunk_total_vector)

                        if result > most_similar_result:
                            most_similar_result = result
                            most_similar_chunk = x

                # cols = ["Sentence", "MostSimilarChunk", "Similarity"]
                row = sheetCalendar.row(outputFileLineIndex)
                row.write(0, current_dsc_sentence.sentence)
                row.write(1, most_similar_chunk)
                row.write(2, most_similar_result)
                if most_similar_result > threshold :
                    row.write(3, 1)
                outputFileLineIndex = outputFileLineIndex + 1

    if (inputFileName != "Read_Contacts") & (1 == 0):
        words, w2i, permissions, applications = Utils.read_whyper_data("Read_Contacts.xls", "excel", True, chunk_gram, False)

        sheetCotacts = outputFile.add_sheet("Read_Contacts")

        cols = ["Sentence", "MostSimilarChunk", "Similarity"]
        txt = "Row %s, Col %s"
        row = sheetCotacts.row(0)
        for index, col in enumerate(cols):
            row.write(index, col)
        outputFileLineIndex = 1

        for current_app in applications:

            for current_dsc_sentence in current_app.dsc_sentences:
                most_similar_result = 0
                most_similar_chunk = ""

                current_chunk_list = current_dsc_sentence.chunk_list
                chunk_total_vector = 0  # initialize
                for x in current_chunk_list:
                    tokenizer = RegexpTokenizer(r'\w+')
                    exists = 0
                    for w in tokenizer.tokenize(x):
                        if Utils.to_lower(w, True) in wiki_vector:

                            if exists == 0:
                                chunk_total_vector = wiki_vector.get(Utils.to_lower(w, True))
                                exists = 1
                            else:
                                current_chunk_vec = wiki_vector.get(Utils.to_lower(w, True))
                                chunk_total_vector = numpy.array(chunk_total_vector) + numpy.array(current_chunk_vec)
                    if exists > 0:
                        result = Utils.cos_similiariy(permissionVector, chunk_total_vector)

                        if result > most_similar_result:
                            most_similar_result = result
                            most_similar_chunk = x

                # cols = ["Sentence", "MostSimilarChunk", "Similarity"]
                row = sheetCotacts.row(outputFileLineIndex)
                row.write(0, current_dsc_sentence.sentence)
                row.write(1, most_similar_chunk)
                row.write(2, most_similar_result)
                outputFileLineIndex = outputFileLineIndex + 1

    if (inputFileName != "Record_Audio") & (1 == 0):
        words, w2i, permissions, applications = Utils.read_whyper_data("Record_Audio.xls", "excel", True, chunk_gram, False)

        sheetAudio = outputFile.add_sheet("Record_Audio")

        cols = ["Sentence", "MostSimilarChunk", "Similarity"]
        txt = "Row %s, Col %s"
        row = sheetAudio.row(0)
        for index, col in enumerate(cols):
            row.write(index, col)
        outputFileLineIndex = 1

        for current_app in applications:

            for current_dsc_sentence in current_app.dsc_sentences:
                most_similar_result = 0
                most_similar_chunk = ""

                current_chunk_list = current_dsc_sentence.chunk_list
                chunk_total_vector = 0  # initialize
                for x in current_chunk_list:
                    tokenizer = RegexpTokenizer(r'\w+')
                    exists = 0
                    for w in tokenizer.tokenize(x):
                        if Utils.to_lower(w, True) in wiki_vector:

                            if exists == 0:
                                chunk_total_vector = wiki_vector.get(Utils.to_lower(w, True))
                                exists = 1
                            else:
                                current_chunk_vec = wiki_vector.get(Utils.to_lower(w, True))
                                chunk_total_vector = numpy.array(chunk_total_vector) + numpy.array(current_chunk_vec)
                    if exists > 0:
                        result = Utils.cos_similiariy(permissionVector, chunk_total_vector)

                        if result > most_similar_result:
                            most_similar_result = result
                            most_similar_chunk = x

                # cols = ["Sentence", "MostSimilarChunk", "Similarity"]
                row = sheetAudio.row(outputFileLineIndex)
                row.write(0, current_dsc_sentence.sentence)
                row.write(1, most_similar_chunk)
                row.write(2, most_similar_result)
                outputFileLineIndex = outputFileLineIndex + 1


    sheet1 = outputFile.add_sheet("Comparison")

    cols = ["Permission", "SI", "TP", "FP", "FN", "TN", "P(%)", "R(%)", "FS(%)", "Acc(%)"]
    txt = "Row %s, Col %s"
    row = sheet1.row(0)
    for index, col in enumerate(cols):
        row.write(index, col)

    cols = ["READ CONTACTS", "204", "186", "18", "49", "2930", "91.2", "79.1", "84.7", "97.9"]
    txt = "Row %s, Col %s"
    row = sheet1.row(1)
    for index, col in enumerate(cols):
        row.write(index, col)

    cols = ["READ CALENDAR", "288", "241", "47", "42", "2422", "83.7", "85.1", "84.4", "96.8"]
    txt = "Row %s, Col %s"
    row = sheet1.row(2)
    for index, col in enumerate(cols):
        row.write(index, col)

    cols = ["RECORD AUDIO", "259", "195", "64", "50", "3470", "75.9", "79.7", "77.4", "97.0"]
    txt = "Row %s, Col %s"
    row = sheet1.row(3)
    for index, col in enumerate(cols):
        row.write(index, col)

    cols = ["TOTAL", "751", "622", "129", "141", "9061", "82.8", "81.5", "82.2", "97.3"]
    txt = "Row %s, Col %s"
    row = sheet1.row(4)
    for index, col in enumerate(cols):
        row.write(index, col)

    outputFile.save(outputFileName)
