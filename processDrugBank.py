import lxml.etree as ET
from tqdm import tqdm
import numpy as np
import pandas as pd


def data_process():
    drugbank_path = "D:/drugbank/full database.xml"
    drugbank = ET.parse(drugbank_path)
    root = drugbank.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    drug_content = []
    drug_network = []
    drug_action = []
    drug_state = []

    # for drug in tqdm(root.xpath("db:drug[db:groups/db:group='approved']", namespaces=ns)):
    for drug in tqdm(root.xpath("db:drug", namespaces=ns)):
        drugbank_id = drug.find("db:drugbank-id[@primary='true']", ns).text
        drugName = drug.find("db:name", ns).text
        drugDescription = drug.find("db:description", ns).text
        if drugDescription is not None:
            drugDescription = drugDescription.replace("\n", "")
            drugDescription = drugDescription.replace("\r", "")
        state = drug.find("db:state", ns)
        if state is not None:
            state = state.text
        else:
            state = "None"
        drugGroup = drug.find("db:groups/db:group", ns).text

        drugIndication = drug.find("db:indication", ns).text
        if drugIndication is not None:
            drugIndication = drugIndication.replace("\n", "")
            drugIndication = drugIndication.replace("\r", "")

        drugIndication = drug.find("db:indication", ns).text
        if drugIndication is not None:
                drugIndication = drugIndication.replace("\n", "")
                drugIndication = drugIndication.replace("\r", "")


        drug_interactions = drug.xpath("db:drug-interactions/db:drug-interaction", namespaces=ns)
        target = drug.xpath("db:targets/db:target", namespaces=ns)
        if len(target) > 0:
            t = target[0]
            action = t.find("db:actions/db:action", ns)
            if action is not None:
                action = action.text
            else:
                action = "None"
        else:
            action = None
        drug_action.append([drugbank_id, action])
        drug_content.append([drugbank_id, drugName, state, drugDescription])
        for t in drug_interactions:
            t_drugbank_id = t.find("db:drugbank-id", ns).text
            name = t.find("db:name", ns).text
            interaction = t.find("db:description", ns).text
            drug_network.append([drugbank_id, t_drugbank_id, name, interaction, "1"])
    drug_action = np.array(drug_action, dtype=np.str)
    np.savetxt("drugs_labels.csv", drug_action, fmt="%s", delimiter=",", encoding="utf-8")

    drug_content = np.array(drug_content, dtype=np.str)
    drug_network = np.array(drug_network, dtype=np.str)
    np.savetxt("drugbank_all.content", drug_content, fmt="%s", delimiter='|', encoding="utf-8")
    np.savetxt("drugbank_all.cites", drug_network, fmt="%s", delimiter=',', encoding="utf-8")
    np.savetxt("drugbank_all_d2d.cites", drug_network[:, [0, 1]], fmt="%s", delimiter=" ", encoding="utf-8")
    np.savetxt("drugbank_all_d2d_LINE.cites", drug_network[:, [0, 1, 4]], fmt="%s", delimiter=" ", encoding="utf-8")



def get_labels():
    data = pd.read_csv("G:\\CEGNN\\materials\\drugbank\\drugbank2.content", delimiter="|",
                       names=["drugbank_id", "drugName", "state", "desc"], index_col=False)
    ids_labels = data[["drugbank_id", "state"]]
    ids_labels.to_csv("G:\\CEGNN\\materials\\drugbank\\drugbank_labels.csv",
                      encoding="utf-8", sep=",", header=False, index=False)


def process_drugbank_attribute():
    """
    提取DrugBank属性特征
    提取关联的蛋白质
    """
    drugbank_path = "D:/drugbank/full database.xml"
    drugbank = ET.parse(drugbank_path)
    root = drugbank.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    drug_attribute = []
    drug_uniprots = []

    for drug in tqdm(root.xpath("db:drug", namespaces=ns)):
        drugName = drug.find("db:name", ns).text
        drugbank_id = drug.find("db:drugbank-id[@primary='true']", ns).text
        cas_number = drug.find("db:cas-number", ns).text
        unii = drug.find("db:unii", ns).text

        state = drug.find("db:state", ns)
        if state is not None:
            state = state.text
        else:
            state = "None"

        drug_interactions = drug.xpath("db:drug-interactions/db:drug-interaction", namespaces=ns)
        interactions_count = len(drug_interactions)

        group = drug.xpath("db:groups/db:group", namespaces=ns)
        group_name = ""
        if group is not None:
            for g in group:
                group_name = group_name + "#" + g.text

        clsf = drug.find("db:classification", ns)
        direct_parent = "None"
        kingdom = "None"
        superclass = "None"
        cclass = "None"
        subclass = "None"
        if clsf is not None:
            direct_parent = clsf.find("db:direct-parent", ns).text
            kingdom = clsf.find("db:kingdom", ns).text
            superclass = clsf.find("db:superclass", ns).text
            cclass = clsf.find("db:class", ns).text
            subclass = clsf.find("db:subclass", ns).text
        drug_attribute.append([drugbank_id, drugName, state,
                               cas_number, unii, interactions_count,
                               group_name, direct_parent, kingdom,
                               superclass, cclass, subclass])
        uniprot_ids = drug.xpath("db:pathways/db:pathway/db:enzymes/db:uniprot-id", namespaces=ns)
        if uniprot_ids is not None:
            for uniprot_id in uniprot_ids:
                drug_uniprots.append([drugbank_id, uniprot_id.text])
    drug_attribute = np.array(drug_attribute, dtype=np.str)
    drug_uniprots = np.array(drug_uniprots, dtype=np.str)

    drug_attribute_df = pd.DataFrame(drug_attribute, columns=["DRUG_ID", "NAME", "STATE",
                                                              "CAS_NUMBER", "UNII", "INTERACTION_COUNT",
                                                              "GROUP_NAME", "DRIECT_PARENT", "KINGDOM",
                                                              "SUPERCLASS", "CCLASS", "SUBCLASS",])
    drug_uniprots_pd = pd.DataFrame(drug_uniprots, columns=["drugbank_id", "uniprot_id"])
    drug_attribute_df.to_csv("drugbank_attribute.csv", sep="\t", encoding="utf-8", index=False)
    drug_uniprots_pd.to_csv("drugbank_uniprots.csv", sep="\t", encoding="utf-8", index=False)


