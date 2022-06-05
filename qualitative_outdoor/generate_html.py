import os
import os.path as osp
import numpy as np


head = '''
<html>
<head>
<style>
td {text-align: center;}
</style>
</head>
<p>  
</p>
<br>
<table border="1">
'''

end = '''
</table>
<br>`
</html>
'''

def writeHTML(out_path, results_dirs):
    f = open(out_path, 'w')
    f.write(head + '\n')
    f.write('<tr>'
            '<td style="background-color:#FFFFFF"> ID </td> '
            '<td style="background-color:#FFFFFF"> Input </td> '
            '<td style="background-color:#FFFFFF"> ConvMPN </td> '
            '<td style="background-color:#FFFFFF"> Exp-cls </td> '
            '<td style="background-color:#FFFFFF"> HAWP </td> '
            '<td style="background-color:#FFFFFF"> LETR </td> '
            '<td style="background-color:#FFFFFF"> HEAT (Ours) </td> '
            '<td style="background-color:#FFFFFF"> G.T.  </td> '
            '</tr>')

    fileids_path = '../data/cities_dataset/valid_list.txt'
    img_base = '../data/cities_dataset/rgb'
    with open(fileids_path) as ff:
        file_ids = ff.readlines()
        file_ids = file_ids[50:]
    file_ids = [file_id.strip() for file_id in file_ids]
    permuted_ids = np.random.permutation(file_ids)
    file_ids = permuted_ids[:100]

    for file_id in file_ids:
        row_str = '<tr>'
        row_str += '<td> {} </td>'.format(file_id)
        row_str += '<td> <img src="{}" width="180"> </td>'.format(os.path.join(img_base, file_id + '.jpg'))
        for dir_idx, result_dir in enumerate(results_dirs):
            pred_filepath = osp.join(result_dir, '{}.png'.format(file_id))
            row_str += '<td> <img src="{}" width="180"> </td>'.format(pred_filepath)
        row_str += '</tr>'
        f.write(row_str + '\n')

    f.write(end + '\n')


if __name__ == '__main__':
    results_dirs = ['svg_images_256/convmpn', 'svg_images_256/exp_cls', 'svg_images_256/hawp', 'svg_images_256/letr', 'svg_images_256/heat', 'svg_images_256/gt']

    writeHTML(out_path='./outdoor_qual.html', results_dirs=results_dirs)
