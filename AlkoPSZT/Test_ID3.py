import Data
import ID3

data = Data.StudentsData()
object = data.objects.loc[14]
id3 = ID3.ID3(data.objects,'Walc', data.attributes)
print(id3.class_count(2,data.objects))
print(len(id3.attribute_value_members('absences',10,data.objects)))
tree = id3.build_tree(None, data.attributes, data.objects)
print(tree)
#print(tree.classify(object))
#print(object)
#object = data.objects.loc[104]
#print(tree.classify(object))
#print(object)
#object = data.objects.loc[600]
#print(tree.classify(object))
#print(object)