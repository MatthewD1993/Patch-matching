#pragma once
#include "string"

const std::string seqtypes[] = {"clean","final"};

const std::string seqs[] = {"alley_1","alley_2","ambush_2","ambush_4","ambush_5","ambush_6","ambush_7","bamboo_1","bamboo_2","bandage_1","bandage_2","cave_2","cave_4","market_2","market_5","market_6",
	"mountain_1","shaman_2", "shaman_3", "sleeping_1","sleeping_2","temple_2","temple_3"};
const int seqLengths[] = {50,50,21,33,50,20,50,50,50,50,50,50,50,50,50,40,49,50,50,50,50,50,50};

//int itestingset[] = {0};
int itrainset[] = {0,2,4,5,6,7,9,11,14,15,16,17,20,21};
std::vector<int> isets [4] = {{0,2,4,5,6,7,9,11,14,15,16,17,20,21},{1,3,8,10,12,13,18,19,22},{3},{0}};//testset

const std::string datapath = "/cdengdata/MPI-Sintel-complete/training/"; //Set this
const std::string evalpath = "/media/serv/data/flowevalANN/";//"/disk1/home/bailer/data/floweval/";


const int _fr = 2;  // image in sequence, set 10 for Grove2, set 23 for problem with circular flow
const std::string seq = "temple_3";//"alley_1"; //Seq in sintell ambush_2 market_5


std::string sintelPath(int type, int seq)
{
  return datapath+seqtypes[type]+"/"+seqs[seq]+"/";
}

std::string sintelFiles(int type, int seq)
{
  return datapath+seqtypes[type]+"/"+seqs[seq]+"/frame_%4.png";
}


std::string sintelOutliers(int type, int seq)
{
  return "/bailerdata/Sintel/outliers/training/"+seqtypes[type]+"/"+seqs[seq]+"/frame_%4";
}

std::string sintelFlowPath(int seq)
{
  return datapath+"/flow/"+seqs[seq]+"/";
}

std::string sintelFlowFiles(int seq)
{
  return datapath+"/flow/"+seqs[seq]+"/frame_%4.flo";
}

std::string sintelOccFiles(int seq)
{
  return datapath+"/occlusions/"+seqs[seq]+"/frame_%4.png";
}




