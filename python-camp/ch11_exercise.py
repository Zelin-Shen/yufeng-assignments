def binary_search(nums, target):
    if not nums:  # 处理空数组
        return -1
    if target < nums[0] or target > nums[-1]:  # 目标超出数组范围
        return -1
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            return mid
    return -1

if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 18, 19, 24, 26, 29, 33, 55, 66]
    res = binary_search(nums, 66)
    print(res)