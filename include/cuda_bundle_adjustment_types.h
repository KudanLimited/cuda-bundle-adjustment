/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
#define __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "optimisable_graph.h"

namespace cuba
{

class StereoEdgeSet : public EdgeSet<3>
{
public:

	StereoEdgeSet() {}
	~StereoEdgeSet() {}

private:

};


class MonoEdgeSet : public EdgeSet<2>
{
public:

	MonoEdgeSet() {}
	~MonoEdgeSet() {}
	
private:

};


} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
