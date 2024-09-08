import json

import numpy as np


def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic, np.bool_)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


class Evaluator:
    def __init__(
        self,
        results,
        AXISTHREHOLD=5,
        ORIGINTHRESHOLD=0.1,
        CAREDCAT=["drawer", "door", "lid"],
        save=False,
        save_path=None,
    ):
        # Define the threshold for the motion axis and motion origin
        # The Axis threshold is in degree
        # The origin threshold is in percentage of the part diagonal length
        self.AXISTHREHOLD = AXISTHREHOLD
        self.ORIGINTHRESHOLD = ORIGINTHRESHOLD
        self.CAREDCAT = CAREDCAT
        self.results = results
        self.save = save
        self.save_path = save_path

    def evaluate(self):
        evaluations = {}
        for model_id in self.results.keys():
            evaluations[model_id] = self._evaluateModel(
                self.results[model_id]["gt"], self.results[model_id]["pred"]
            )
        self.evaluations = evaluations

    def summarize(self):
        # Summarize the evaluation results
        # Summarize in the micro-average way and macro-average way
        # micro_average: average over all the parts
        eval_M_micro = []
        eval_MA_micro = []
        eval_MAO_micro = []
        axis_diff_micro = []
        origin_diff_micro = []
        # micro_average, but only consider a specific part category
        eval_M_micro_cat = {}
        eval_MA_micro_cat = {}
        eval_MAO_micro_cat = {}
        axis_diff_micro_cat = {}
        origin_diff_micro_cat = {}
        for cat in self.CAREDCAT:
            eval_M_micro_cat[cat] = []
            eval_MA_micro_cat[cat] = []
            eval_MAO_micro_cat[cat] = []
            axis_diff_micro_cat[cat] = []
            origin_diff_micro_cat[cat] = []
        # macro_average: average over all the models (This macro is averaged over the models)
        eval_M_macro = []
        eval_MA_macro = []
        eval_MAO_macro = []
        # MODEL Performance
        model_performance = {}
        for model_id in self.evaluations.keys():
            eval_M_macro_model = []
            eval_MA_macro_model = []
            eval_MAO_macro_model = []
            part_evaluations = self.evaluations[model_id]
            for part_evaluation in part_evaluations:
                eval_M_macro_model.append(part_evaluation["eval_M"])
                # Add the category-based evaluation
                eval_M_micro_cat[part_evaluation["cat"]].append(
                    part_evaluation["eval_M"]
                )
                if part_evaluation["gt_motionType"] == "fixed":
                    # Actually our CAREDCAT does not include the fixed parts, this code is just for debugging
                    import pdb
                    pdb.set_trace()
                    eval_MA_macro_model.append(part_evaluation["eval_M"])
                    eval_MAO_macro_model.append(part_evaluation["eval_M"])
                else:
                    axis_diff_micro.append(part_evaluation["axis_diff"])
                    eval_MA_macro_model.append(
                        part_evaluation["eval_M"] and part_evaluation["eval_A"]
                    )
                    # Add the category-based evaluation
                    axis_diff_micro_cat[part_evaluation["cat"]].append(part_evaluation["axis_diff"])
                    eval_MA_micro_cat[part_evaluation["cat"]].append(
                        part_evaluation["eval_M"] and part_evaluation["eval_A"]
                    )
                    if part_evaluation["gt_motionType"] == "prismatic":
                        eval_MAO_macro_model.append(
                            part_evaluation["eval_M"] and part_evaluation["eval_A"]
                        )
                        # Add the category-based evaluation
                        eval_MAO_micro_cat[part_evaluation["cat"]].append(
                            part_evaluation["eval_M"] and part_evaluation["eval_A"]
                        )
                    else:
                        origin_diff_micro.append(part_evaluation["origin_diff"])
                        eval_MAO_macro_model.append(
                            part_evaluation["eval_M"]
                            and part_evaluation["eval_A"]
                            and part_evaluation["eval_O"]
                        )
                        # Add the category-based evaluation
                        origin_diff_micro_cat[part_evaluation["cat"]].append(part_evaluation["origin_diff"])
                        eval_MAO_micro_cat[part_evaluation["cat"]].append(
                            part_evaluation["eval_M"]
                            and part_evaluation["eval_A"]
                            and part_evaluation["eval_O"]
                        )
            if len(eval_M_macro_model) == 0:
                continue
            # Update the micro results
            eval_M_micro += eval_M_macro_model
            eval_MA_micro += eval_MA_macro_model
            eval_MAO_micro += eval_MAO_macro_model
            # Update the macro results
            eval_M_macro.append(np.mean(eval_M_macro_model))
            eval_MA_macro.append(np.mean(eval_MA_macro_model))
            eval_MAO_macro.append(np.mean(eval_MAO_macro_model))
            # Record the model performance for visualization sorting
            model_performance[model_id] = {
                "eval_M": np.mean(eval_M_macro_model),
                "eval_MA": np.mean(eval_MA_macro_model),
                "eval_MAO": np.mean(eval_MAO_macro_model),
            }
        # Recrod the micro results
        performance = {}
        performance["micro_M"] = np.mean(eval_M_micro)
        performance["micro_MA"] = np.mean(eval_MA_micro)
        performance["micro_MAO"] = np.mean(eval_MAO_micro)
        performance["micro_axis_diff"] = np.mean(axis_diff_micro)
        performance["micro_origin_diff"] = np.mean(origin_diff_micro)
        # Record the micro results on each part category
        performance["part_num"] = {}
        performance["micro_M_cat"] = {}
        performance["micro_MA_cat"] = {}
        performance["micro_MAO_cat"] = {}
        performance["micro_axis_diff_cat"] = {}
        performance["micro_origin_diff_cat"] = {}
        for cat in self.CAREDCAT:
            performance["part_num"][cat] = len(eval_M_micro_cat[cat])
            performance["micro_M_cat"][cat] = np.mean(eval_M_micro_cat[cat])
            performance["micro_MA_cat"][cat] = np.mean(eval_MA_micro_cat[cat])
            performance["micro_MAO_cat"][cat] = np.mean(eval_MAO_micro_cat[cat])
            performance["micro_axis_diff_cat"][cat] = np.mean(axis_diff_micro_cat[cat])
            if len(origin_diff_micro_cat[cat]) == 0:
                performance["micro_origin_diff_cat"][cat] = -1
            else:
                performance["micro_origin_diff_cat"][cat] = np.mean(origin_diff_micro_cat[cat])
        # Record the macro results
        performance["macro_M"] = np.mean(eval_M_macro)
        performance["macro_MA"] = np.mean(eval_MA_macro)
        performance["macro_MAO"] = np.mean(eval_MAO_macro)

        if self.save:
            evaluation_results = {}
            evaluation_results["performance"] = performance
            evaluation_results["model_performance"] = model_performance
            evaluation_results["evaluations"] = self.evaluations
            print(f"Saving the evaluation results to {self.save_path}")
            with open(self.save_path, "w") as f:
                evaluation_results = convert_numpy_to_python(evaluation_results)
                json.dump(evaluation_results, f)

        return performance

    def _evaluateModel(self, gt_parts, pred_parts):
        # Evaluate the articulated model (only evaluate on the parts that are cared)
        part_ids = gt_parts.keys()
        part_evaluations = []
        for part_id in part_ids:
            gt_part = gt_parts[part_id]
            pred_part = pred_parts[part_id]
            # These stuffs should be the same
            assert gt_part["id"] == pred_part["id"]
            assert gt_part["cat"] == pred_part["cat"]
            if gt_part["cat"] not in self.CAREDCAT:
                # Ignore the parts that are not cared
                continue
            part_evaluations.append(self._evaluatePart(gt_part, pred_part))
        return part_evaluations

    def _evaluatePart(self, gt_part, pred_part):
        part_evaluation = {
            "part_id": gt_part["id"],
            "cat": gt_part["cat"],
            "gt_parent": gt_part["parent"],
            "gt_motionType": gt_part["motionType"],
            "gt_motionAxis": gt_part["motionAxis"],
            "gt_motionOrigin": gt_part["motionOrigin"],
            "pred_parent": pred_part["parent"],
            "pred_motionType": pred_part["motionType"],
            "pred_motionAxis": pred_part["motionAxis"],
            "pred_motionOrigin": pred_part["motionOrigin"],
            "gt_id": gt_part["gt_id"]
        }
        # Evaluate the motion type
        gt_motion_type = gt_part["motionType"]
        pred_motion_type = pred_part["motionType"]
        eval_M = gt_motion_type == pred_motion_type
        part_evaluation["eval_M"] = bool(eval_M)
        # Evaluate the motion axis
        gt_motion_axis = np.asarray(gt_part["motionAxis"])
        pred_motion_axis = np.asarray(pred_part["motionAxis"])
        axis_diff = self._evaluateAxis(gt_motion_axis, pred_motion_axis)
        eval_A = axis_diff < self.AXISTHREHOLD
        part_evaluation["axis_diff"] = axis_diff
        part_evaluation["eval_A"] = bool(eval_A)
        # Evaluate the motion origin
        gt_motion_origin = np.asarray(gt_part["motionOrigin"])
        pred_motion_origin = np.asarray(pred_part["motionOrigin"])
        origin_diff = self._evaluateOrigin(
            gt_motion_origin, pred_motion_origin, gt_motion_axis, gt_part["diagonal"]
        )
        eval_O = origin_diff < self.ORIGINTHRESHOLD
        part_evaluation["origin_diff"] = origin_diff
        
        part_evaluation["eval_O"] = bool(eval_O)

        return part_evaluation

    def _evaluateAxis(self, gt_axis, pred_axis):
        # Calculate the difference between the pred and gt axis (0-90 degree)
        axis_diff = np.dot(gt_axis, pred_axis) / (
            np.linalg.norm(gt_axis) * np.linalg.norm(pred_axis)
        )
        if axis_diff < 0:
            axis_diff = -axis_diff
        axis_diff = min(axis_diff, 1.0)
        axis_diff = np.arccos(axis_diff) / np.pi * 180

        return axis_diff

    def _evaluateOrigin(self, gt_origin, pred_origin, gt_axis, diagonal):
        # Distance between the predicted origin and the ground truth axis line (meter)
        p = pred_origin - gt_origin
        origin_diff = (
            np.linalg.norm(np.cross(p, gt_axis)) / np.linalg.norm(gt_axis) / diagonal
        )
        return origin_diff
