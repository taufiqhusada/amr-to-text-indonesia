import os

source_file = 'generated_summary_amr.txt'
with open(source_file, encoding='utf8') as f:
    source_data = f.readlines()

target_folder = 'graf_ringkasan/'
outfile = None
for line in source_data:
    if ('# ::id' in line):
        if (outfile):
            outfile.close()
    
        filename = line.strip().split('# ::id ')[1] + '.txt'
        outfile = open(os.path.join(target_folder, filename),  'w+', encoding='utf8' )
        
    outfile.write(line)
outfile.close()
