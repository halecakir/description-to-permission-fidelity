import xlrd
import xlwt


def modify_whyper_data(file_path, out_file):
    loc = (file_path)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    file_name = file_path.split("/")[-1]
    if file_name == "Read_Contacts.xls":
        data = []
        for r in range(sheet.nrows):
            row = []
            for c in range(sheet.ncols):
                row.append(sheet.cell_value(r, c))
            data.append(row)

            sentence = sheet.cell_value(r,0)
            if sentence.startswith("#"):
                copy_row = [c for c in row]
                copy_row[0] = row[1] 
                data.append(copy_row)

        workbook = xlwt.Workbook()
        new_sheet = workbook.add_sheet('test')
        for row_index, row_values in enumerate(data):
            del row_values[1] #remove unnecessary item
            for col_index, cell_value in enumerate(row_values):
                
                new_sheet.write(row_index, col_index, cell_value)
        workbook.save(out_file)


if __name__=="__main__":              
    modify_whyper_data("/home/huseyin/Desktop/Security/data/whyper/Read_Contacts.xls",
                        "/home/huseyin/Desktop/Security/data/whyper/Read_Contacts_modified.xls") 
